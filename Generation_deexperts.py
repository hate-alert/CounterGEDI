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


from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


params = {
    'sep_token':'<|endoftext|>',
    'max_generation_length': 200,
    'min_generation_length':40,
    'max_input_length':128,
    'num_beams':5,
    'no_repeat_ngram_size':5,
    'repition_penalty': 3.5,
    'k':100,
    'p':0.92,
    'filter_p':0.9,
    'sample':True,
    'temperature':1.2,
    'early_stopping':True,
    'model_path':'gpt2-medium',
    'dataset_hate':'CONAN',
    'task_name':[('Politeness','polite'),('Toxicity','toxic')],
    'coefficient':[3.2,-3.2],
    'save_path': './../HULK/Counterspeech/Results/',
    'device': 'cuda',
    'batch_size':4,
    'cache_path':'../HULK/Saved_models/'
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

def generate(params,hate_sentences,model,controller_list,tokenizer,device):
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
                
                
                
                if(len(controller_list)>0):
                    controller_logits=[]
                    for model_temp in controller_list:
                        temp_outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
                        logits_temp=temp_outputs.logits
                        controller_logits.append(logits_temp)
                    for i in range(len(controller_list)):
                        alpha = torch.tensor(params['coefficient'][i]).to(device)
                        ensemble_logits += alpha * (controller_logits[i])

                
                
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
               
        
            cntr += decoded_outputs
    print(len(cntr))
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
    refs = [mp[ht_i] for ht_i in hate]
    return hate,refs             # a given hate instance and refrences(replies) for metrics evaluation




def main(params,model_path,dataset):
    path_models   = '../HULK/Counterspeech/Saved_Models/Generator'
    path_datasets = '../HULK/Counterspeech/Datasets'

    
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
#             deviceID = get_gpu()
            torch.cuda.set_device(1)
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
    for task in params['task_name']:
        path_model_task=path_models+'/'+task[0]+'_dexpert_gpt2_'+task[1]+'/'
        model_temp=AutoModelForCausalLM.from_pretrained(path_model_task,cache_dir=cache_path)
        model_temp.to(device)
        model_temp.eval()
        controller_list.append(model_temp)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path,cache_dir=cache_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,cache_dir=cache_path)
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
    cntr_replies = generate(params,hate,model,controller_list,tokenizer,device)
    
    
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
        hate_counter_replies[count]=temp
        count+=1

    dict_results['samples']=hate_counter_replies  
    
    model_path_modified = "-".join(model_path.split('/')[-2:])
    print(model_path_modified)
    write_in=params["save_path"] + model_path_modified +"_on_"+dataset+"_dexpert.json"
    with open(write_in, 'w') as outfile:
         json.dump(dict_results, outfile,indent=4)
    
    
                         
    
if __name__ == "__main__":
    
    saved_path='../HULK/Counterspeech/Saved_Models/Generator/'
    
    
#     model_paths=["microsoft/DialoGPT-medium", "gpt2-medium", 
#                 saved_path+'Reddit_DialoGPT-medium', saved_path+'Gab_DialoGPT-medium',
#                 saved_path+'CONAN_DialoGPT-medium' , saved_path+'Create_debate_DialoGPT-medium']
    model_paths=[saved_path+'Create_debate_DialoGPT-medium']
    datasets = ["CONAN", "Reddit", "Gab"]
    
    total = [model_paths, datasets]
    for element in itertools.product(*total):
        model=element[0]
        dataset=element[1]
        print(model,dataset)
        main(params,model,dataset)
    
    
    
    






