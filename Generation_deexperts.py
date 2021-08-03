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
import itertools
from Generation.eval import *
from Generation.utils import *
from Generation.models import *

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

params = {
    'sep_token':'<|endoftext|>',
    'max_generation_length': 200,
    'min_generation_length':40,
    'max_input_length':128,
    'num_beams':1,
    'no_repeat_ngram_size':5,
    'repitition_penalty': 3.5,
    'control_type':'gedi',
    'k':100,
    'p':0.92,
    'filter_p':0.8,
    'target_p':0.8,
    'disc_weight':30,
    'class_bias':0,
    'sample':True,
    'temperature':1.2,
    'early_stopping':True,
    'model_path':'gpt2-medium',
    'dataset_hate':'CONAN',
    'task_name':[('Emotion','joy')],
    'coefficient':[4.5],
    'save_path': './../HULK_new/Counterspeech/Results/',
    'device': 'cuda',
    'batch_size':4,
    'cache_path':'./../HULK_new/Saved_models/',
    'generation_method':'huggingface',
    'class_bias':0,
}



#task_name  is a list having element in the format (task_name,class_name)
def get_gpu():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 1.0, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)



# In[23]:
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


def generate_huggingface_method(params,hate_sentences,model,controller_list,tokenizer,device,control_type='dexpert'):
    cntr = []
    model.eval()
    
    alpha_controller=[]
    for i in range(len(params['coefficient'])):
        alpha = torch.tensor(params['coefficient'][i]).to(device)
        alpha_controller.append(alpha)
    print(alpha_controller)
    for step in tqdm(range(len(hate_sentences))):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        input_ids = tokenizer.encode(hate_sentences[step],truncation=True,max_length=params['max_input_length'],return_tensors='pt') 
        eos = tokenizer.encode(params['sep_token'],return_tensors='pt')
        input_ids = torch.cat((input_ids,eos),1)
        input_ids=input_ids.to(device)
        ####### greedy_Decoding ######
        beam_outputs = model.generate(
            controller_alphas=alpha_controller,
            controller_list=controller_list,
            control_type=control_type,
            positive_class='true',
            negative_class='false',
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
        reply = (tokenizer.decode(beam_outputs[0])).split(params['sep_token'])[1]
        print("hate",hate_sentences[step])
        print("counter",reply)
        cntr.append(reply)
        if step>0 and step%100==0:
            print("doing")
    return cntr


def generate_single_own(params,hate_sentences,model,controller_list,tokenizer,device,use_control=True):
    
    cntr=[]
    for step in tqdm(range(len(hate_sentences))):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        input_ids = tokenizer.encode(hate_sentences[step],truncation=True,max_length=params['max_input_length'],return_tensors='pt') 
        eos = tokenizer.encode(params['sep_token'],return_tensors='pt')
        input_ids = torch.cat((input_ids,eos),1)
        input_ids=input_ids.to(device)
        with torch.no_grad():
            for step in range(params['max_generation_length']):
                outputs = model(input_ids)
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
                        temp_outputs = model(input_ids)
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

                
                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                
                if next_tokens == tokenizer.eos_token_id:
                    break
                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
            reply = (tokenizer.decode(input_ids[0])).split(params['sep_token'])[1]   
            cntr.append(reply)
    return cntr

    

def hate_refrences(data,test_set):
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




def main(params,model_path,dataset):
    path_models   = './../HULK_new/Counterspeech/Saved_Models/Generator'
    path_models_disc   = './../HULK_new/Counterspeech/Saved_Models/Discriminator'
    path_datasets = './../HULK_new/Counterspeech/Datasets'

    
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
#             deviceID = get_gpu()
            torch.cuda.set_device(0)
            ##### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
#             torch.cuda.set_device(args.gpuid)

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    
    
    test_path  = path_datasets+'/'+dataset+'/Test.csv'
    train_path  = path_datasets+'/'+dataset+'/Train.csv'

    cache_path = params['cache_path']
    
    #Politeness_dexpert_gpt2_polite
    controller_list=[]
    if(params['control_type']=='dexpert'):
        for task in params['task_name']:
            path_model_task=path_models+'/'+task[0]+'_dexpert_gpt2_'+task[1]+'/'
            model_temp=AutoModelForCausalLM.from_pretrained(path_model_task,cache_dir=cache_path)
            model_temp.to(device)
            model_temp.eval()
            controller_list.append(model_temp)
    elif(params['control_type']=='gedi'):
        if(len(params['task_name'])>1):
            print("gedi model controls attribute one at a time")
        else:
            print("loaded_gedi")
            task =params['task_name'][0]
            path_model_task=path_models_disc+'/'+task[0]+'_gedi_gpt2_'+task[1]+'/'
            model_temp=Model_Generation.from_pretrained(path_model_task,cache_dir=cache_path)
            model_temp.to(device)
            model_temp.eval()
            controller_list.append(model_temp)
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
    model = Model_Generation.from_pretrained(model_path,cache_dir=cache_path)
    if(params['generation_method']=='own'):
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    
    model.to(device)
    model.eval()
    
    test  = pd.read_csv(test_path)
    
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    train_set = list(zip(train['initiator_message'].tolist(),train['reply_message'].tolist()))
    test_set  = list(zip(test['initiator_message'].tolist(),test['reply_message'].tolist()))
    data = []
    for x in test_set:
        data.append(x)
    for x in train_set:
        data.append(x)
    print(len(data),len(test_set),len(train_set))
    hate, cntr_refs = hate_refrences(data,test_set) 
    
    
    if(params['generation_method']=='huggingface'):
        cntr_replies = generate_huggingface_method(params,hate,model,controller_list,tokenizer,device,control_type=params['control_type'])
    elif(params['generation_method']=='own'):
        cntr_replies = generate_single_own(params,hate,model,controller_list,tokenizer,device,use_control=True)
    
    
    del model
    for controller in controller_list:
        del controller
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
        print(temp)
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
    
    
    if(params['generation_method']=='huggingface'):
        if(params['control_type']=='dexpert'):
            write_in=params["save_path"] + model_path_modified +"_on_"+dataset+"_dexpert_huggingface_"+ts+".json"
        elif(params['control_type']=='gedi'):
            write_in=params["save_path"] + model_path_modified +"_on_"+dataset+"_gedi_huggingface_"+ts+".json"
    elif(params['generation_method']=='own'):
        write_in=params["save_path"] + model_path_modified +"_on_"+dataset+"_dexpert_"+ts+".json"

    with open(write_in, 'w') as outfile:
         json.dump(dict_results, outfile,indent=4)
    
    
                         
    
if __name__ == "__main__":
    
    saved_path='../HULK_new/Counterspeech/Saved_Models/Generator/'
    
    
    #model_paths=[saved_path+'Reddit_DialoGPT-medium', saved_path+'Gab_DialoGPT-medium',
    #saved_path+'CONAN_DialoGPT-medium' , saved_path+'Create_debate_DialoGPT-medium']
    #model_paths=[saved_path+'CONAN_DialoGPT-medium']
    model_paths=['microsoft/DialoGPT-medium']
    datasets = ["CONAN","Reddit","Gab"]
    
    total = [model_paths, datasets]
    for element in itertools.product(*total):
        model=element[0]
        dataset=element[1]
        print(model,dataset)
        main(params,model,dataset)
    
    
    
    






