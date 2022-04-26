import json
from os import listdir
import glob
from tqdm import tqdm
import transformers 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd
import numpy as np
import GPUtil
import time
from Generation.eval import *
from Generation.utils import *
from Generation.models import *
import argparse

debug=False
# from transformers import (
#     MODEL_WITH_LM_HEAD_MAPPING,
#     WEIGHTS_NAME,
#     AdamW,
#     AutoConfig,
#     AutoModelWithLMHead,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     PreTrainedModel,
#     PreTrainedTokenizer,
#     get_linear_schedule_with_warmup,
# )

from transformers import AutoTokenizer,AutoModelForCausalLM
HULK_path='../HULK/' 

print(HULK_path)


#task_name  is a list having element in the format (task_name,class_name)
def get_gpu(gpu_id):
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 2, maxLoad = 1.0, maxMemory = 0.7, includeNan=False, excludeID=[], excludeUUID=[])
        for i in range(len(tempID)):
            if len(tempID) > 0 and (tempID[i]==gpu_id):
                print("Found a gpu")
                print('We will use the GPU:',tempID[i],torch.cuda.get_device_name(tempID[i]))
                deviceID=[tempID[i]]
                return deviceID
            else:
                time.sleep(5)

def get_dataloader(sentences, tokenizer, params):
    sents=[]
    attns=[]
    
    inputs = tokenizer(sentences,truncation=True,max_length=params['max_input_length'], padding=True)
    for element in inputs['input_ids']:
        sents.append(element+[50256])
    for element in inputs['attention_mask']:
        attns.append(element+[1])
    sents=torch.tensor(sents)
    attns=torch.tensor(attns)
    data = TensorDataset(sents, attns)
    sampler = SequentialSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=params['batch_size'])

def generate(params,hate_sentences,model,controller_list,tokenizer,device,use_control=True):
    cntr = []
    test_dataloader=get_dataloader(hate_sentences, tokenizer,params)
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        batch_size, input_seq_len = input_ids.shape
        position_ids = attention_mask.cumsum(dim=1) - 1
        
        position_ids = (attention_mask.long().cumsum(-1) - 1)
        position_ids.masked_fill_(attention_mask==0, 0) # can be filled with anything >= 0
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=device)
        with torch.no_grad():
            for step in range(params['max_generation_length']):
                outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
                ensemble_logits=outputs.logits
                
                if params['filter_p'] < 1.0:
                    ensemble_logits = top_k_top_p_filtering(ensemble_logits, top_p=params['filter_p'])
                if(debug==True):
                    print("before controller")
                    print(ensemble_logits[:,-1,:])
                    print(torch.max(ensemble_logits[:,-1,:], dim=-1))

                if(len(controller_list)>0 and use_control):
                    controller_logits=[]
                    for model_temp in controller_list:
                        temp_outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
                        logits_temp=temp_outputs.logits
                        #print(logits_temp)
                        controller_logits.append(logits_temp)
                    for i in range(len(controller_list)):
                        alpha = torch.tensor(params['coefficient'][i]).to(device)
                        ensemble_logits += alpha * (controller_logits[i])
                if(debug==True):
                    print("after controller")
                    print(ensemble_logits[:,-1,:])
                    print(torch.max(ensemble_logits[:,-1,:], dim=-1))

                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = ensemble_logits[:, -1, :]

                if params['sample']==True:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if params['temperature'] != 1.0:
                        next_token_logits = next_token_logits /  params['temperature']
                    
                    # Top-p/top-k filtering
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=params['k'], top_p=params['p'])
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
               
            print(decoded_outputs)
            cntr += decoded_outputs
    print(len(cntr))
    return cntr


def generate_huggingface_method(params,hate_sentences,model,controller_list,tokenizer,device,control_type='gedi',num_samples=10):
    """
    Generates senetenecs using Huggingface Generation Method
    """
    cntr = []
    model.eval()
    
    alpha_controller=[]
    for i in range(len(params['coefficient'])):
        alpha = torch.tensor(params['coefficient'][i]).to(device)
        alpha_controller.append(alpha)
    print(alpha_controller)
    for step in tqdm(range(len(hate_sentences))):
        cntr_temp=[]
        for i in range(num_samples):
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            input_ids = tokenizer.encode(hate_sentences[step],truncation=True,max_length=params['max_input_length'],return_tensors='pt') 
            eos = tokenizer.encode(params['sep_token'],return_tensors='pt')
            input_ids = torch.cat((input_ids,eos),1)
            input_ids=input_ids.to(device)
            
            # input_ids = [Hatespeech]<|endoftext|>

            ####### greedy_Decoding ######
            beam_outputs = model.generate(
                controller_alphas=alpha_controller,
                controller_list=controller_list,
                control_type=control_type,
                positive_class=['false','false','true'],     # Positive class for each Counter-Gedi Model provided in params['task_name'] list
                                                             # towards which the model will move the Generated Logits to (and away from negative class)
                negative_class=['true','true','false'], 
                unpertubed_count=params['unpertubed_count'],
                tokenizer=tokenizer,
                class_bias=params['class_bias'],
                filter_p=params['filter_p'],
                target_p=params['target_p'],
                disc_weight=params['disc_weight'],
                input_ids=input_ids, 
                pad_token_id         = tokenizer.eos_token_id,
                max_length           = params['max_generation_length']+len(input_ids[0]),
                min_length           = params['min_generation_length']+len(input_ids[0]),
                top_k                = params["k"],
                top_p                = params["p"],
                repetition_penalty   = params["repitition_penalty"],
                temperature          = params["temperature"],
                num_beams            = params['num_beams'], 
                do_sample            = params['sample'],
                no_repeat_ngram_size = params['no_repeat_ngram_size'],  
                early_stopping       = params['early_stopping']
            )

            # beam_output = [Hatespeech]<|endoftext|>[Generated Counterspeech]
            reply = (tokenizer.decode(beam_outputs[0])).split(params['sep_token'])[1]
            cntr_temp.append(reply)
        print("hate",hate_sentences[step])
        print("counter",cntr_temp[0])
        cntr.append(cntr_temp)
        if step>0 and step%100==0:
            print("doing")
    return cntr
    
def hate_refrences(data,test_set):
    """
    Extract all counterspeech references for each hatespeech in the given dataset
    """
    hate  = []
    reply = []
    refrences = []
    for sample in data:
        ht , rep = sample[0] , sample[1]
        hate.append(ht)
        reply.append(rep)
    hate = list(set(hate))
    mp={}
    for ht_i in hate:
        refs = []
        for sample in data:
            ht_j , rep =  sample[0] , sample[1]
            if ht_j == ht_i:
                refs.append(rep)
        mp[ht_i] = refs
        refrences.append(refs)
    hate = list(set([x[0] for x in test_set]))
    hate.sort(key=lambda s: len(s))
    refs = [mp[ht_i] for ht_i in hate]
    return hate,refs             # a given hate instance and refrences(replies) for metrics evaluation

def main(params,model_path,dataset,gpu_id,num_samples):
    # print(HULK_path)  
    path_models   = HULK_path+'Counterspeech/Saved_Models/Generator'                # Path to Generator Models 
    path_models_disc   = HULK_path+'Counterspeech/Saved_Models/Discriminator'       # Path to Disciminator Gedi Models
    path_datasets = HULK_path+'/Counterspeech/Datasets'

    
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
            deviceID =get_gpu(gpu_id)
#            torch.cuda.set_device(1)
            ##### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            torch.cuda.set_device(deviceID[0])

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    test_path  = path_datasets+'/'+dataset+'/Test.csv'
    train_path  = path_datasets+'/'+dataset+'/Train.csv'

    cache_path = params['cache_path']
    
    #Politeness_dexpert_gpt2_polite
    controller_list=[]
    if(params['control_type']=='gedi'):     # Load controller Gedi models for Generation as per the list provided 
        #if(len(params['task_name'])>1):
        #    print("gedi model controls attribute one at a time")
        #else:
        print("loaded_gedi")
        for task in params['task_name']:
            path_model_task=path_models_disc+'/'+task[0]+'_gedi_gpt2_'+task[1]+'/'
            print(path_model_task)
            model_temp=Model_Generation.from_pretrained(path_model_task,cache_dir=cache_path)
            model_temp.to(device)
            model_temp.eval()
            controller_list.append(model_temp)

    print("Length of Controller List: ", len(controller_list))
    
    # Load saved model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
    model = Model_Generation.from_pretrained(model_path,cache_dir=cache_path)

    model.to(device)
    model.eval()
    
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)[:100]        # Generate counterspeech for 100 test samples, can be modified to entire files
    train_set = list(zip(train['initiator_message'].tolist(),train['reply_message'].tolist()))
    test_set  = list(zip(test['initiator_message'].tolist(),test['reply_message'].tolist()))
    data = []
    for x in test_set:
        data.append(x)
    for x in train_set:
        data.append(x)
    print(len(data),len(test_set),len(train_set))
    hate, cntr_refs = hate_refrences(data,test_set)     # Load all Hate Sentences and a list of Counterspeech References corresponding to that Sentence     
    
    if(params['generation_method']=='huggingface'):
        cntr_replies = generate_huggingface_method(params,hate,model,controller_list,tokenizer,device,control_type=params['control_type'],num_samples=num_samples)
    
    if(params['control_type']!='none'):
        for controller in controller_list:
            del controller
    del model
    torch.cuda.empty_cache()
    
    dict_results={}
    dict_results['params']=params
    hate_counter_replies={}
    
    count=0
    for hate,counter in zip(hate,cntr_replies):
        temp={
            'hatespeech':hate,
            'counterspeech_model':counter,
        }
#         print(temp)
        hate_counter_replies[count]=temp
        count+=1

    dict_results['samples']=hate_counter_replies  
    
    model_path_modified = "-".join(model_path.split('/')[-2:])
    print(model_path_modified)
    
    # using datetime module
    import datetime;
    # ct stores current time
    ct = datetime.datetime.now()
    
    # ts store timestamp of current time
    ts = ct.timestamp()
    ts = str(int(ts))
    
    
    if(params['control_type']=='gedi'):
        write_in=params["save_path"] + model_path_modified +"_on_"+dataset+"_gedi_huggingface_"+ts+"_multi.json"
    else:
        write_in=params["save_path"] + model_path_modified +"_on_"+dataset+"_huggingface_"+ts+"_base.json"
    
    # Dump results into the "write_in" file
    with open(write_in, 'w') as outfile:
         json.dump(dict_results, outfile,indent=4)

params = {
    'sep_token':'<|endoftext|>',
    'max_generation_length': 100,   # Maximum length of Generated Sentence
    'min_generation_length':40,     # Minimum length of Generated Sentence
    'max_input_length':128,         # Maximum length of Input Sentence, after which input is truncated 
    'num_beams':1,
    'unpertubed_count':10,          # Number of tokens to be generated without any perturbation by CounterGedi Model (Only base Model used)
    'no_repeat_ngram_size':5,
    'repitition_penalty': 3.5,
    'control_type':'gedi',          # Takes Values: 'gedi'-> for counterGedi or 'none' (for base model sampling)
    'k':100,
    'p':0.92,
    'filter_p':0.8,
    'target_p':0.8,
    'disc_weight':[0.4,0.3,0.3],    # Weights of different conditionals (controlling parameters) we want to focus on, during generation
    'class_bias':0,
    'sample':True,
    'temperature':1.2,
    'early_stopping':True,
    'model_path':'gpt2-medium',
    'dataset_hate':'CONAN',
    'task_name':[('Emotion', 'joy'),('Politeness', 'polite'),('Toxicity', 'toxic')],    # Controlling parameters, which we want to control while generation
                                                                                        # Used to load the respective Gedi Models for steering of results
    'coefficient':[4.5],
    'save_path': HULK_path+'Counterspeech/Results_new/',        # Path where results are to be stored in json file
    'device': 'cuda',
    'batch_size':4,
    'cache_path':HULK_path+'Saved_models/',
    'generation_method':'huggingface',
    'gpu_id':0
}
                       
    
if __name__ == "__main__":
    
    saved_path=HULK_path+'Counterspeech/Saved_Models/Generator/'        # Path to the folder where the Counter Gedi model is saved
    # model_paths consist of list of trained model folders which contain the respective .pt files of saved models
    model_paths=[saved_path+'CONAN_DialoGPT-medium',saved_path+'Reddit_DialoGPT-medium', saved_path+'Gab_DialoGPT-medium']  
    datasets = ["CONAN","Reddit","Gab"]     # List of Hatespeech-Counterspeech datasets for which we want to Generate counterspeech samples 
    num_samples=5   # Number of samples to be generated per Hatespeech
    for element in zip(model_paths[2:3],datasets[2:3]):
        model=element[0]
        dataset=element[1]
        print(model,dataset)
        main(params,model,dataset,params['gpu_id'],num_samples)
    
    
    
    






