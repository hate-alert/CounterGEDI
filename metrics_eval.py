#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
path_models   = '/home/adarsh-binny' + '/HULK/Counterspeech/Saved_Models/Generator'
path_datasets = '/home/adarsh-binny' + '/HULK/Counterspeech/Datasets'


# In[2]:


from Generation.eval import *
from Generation.generate_parallel import *
import json
from os import listdir
models   = listdir(path_models)
datasets = listdir(path_datasets)

import glob
from tqdm import tqdm

# In[3]:


import transformers 
import torch
import pandas as pd
import numpy as np
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


# In[4]:


args = {
    'sep_token':'<|endoftext|>',
    'max_length': 100,
    'min_length':40,
    'num_beams':5,
    'no_repeat_ngram_size':5,
    'repition_penalty': 3.5,
    'k':100,
    'p':0.92,
    'temperature':1.2,
    'early_stopping':True,
    'model_type':'gpt2-medium',
    'model_name_or_path':'gpt2-medium',
    'config_name':'gpt2-medium',
    'tokenizer_name':'gpt2-medium',
    'Save_path': './../HULK/Counterspeech/Results/'
}


# In[23]:


def beam_search(hate,model,tokenizer):
    cntr = []
    model.eval()
    for step in tqdm(range(len(hate))):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        input_ids = tokenizer.encode(hate[step],truncation=True,max_length=256,return_tensors='pt') 
        eos = tokenizer.encode(args['sep_token'],return_tensors='pt')
        input_ids = torch.cat((input_ids,eos),1)
        
        ####### greedy_Decoding ######
        beam_outputs = model.generate(

            input_ids, 
            pad_token_id         = tokenizer.eos_token_id,
            max_length           = args['max_length']+len(input_ids[0]),
            min_length           = args['min_length']+len(input_ids[0]),
            top_k                = args["k"],
            top_p                = args["p"],
#            repetition_penalty   = args["repetition_penalty"],
            temperature          = args["temperature"],
#             num_beams            = args['num_beans'], 
             no_repeat_ngram_size = args['no_repeat_ngram_size'],  
#             early_stopping       = args['early_stopping']
        )
        reply = (tokenizer.decode(beam_outputs[0])).split(args['sep_token'])[1]
        cntr.append(reply)
        if step>0 and step%100==0:
            print("doing")
    return cntr


# In[6]:



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


# In[7]:


def training_corpus(train_set):
    replies = []
    for sample in train_set:
        rep = sample[1]
        replies.append(rep)
    replies = list(set(replies))
    return replies                # returns the sentences used while training 


 

# In[8]:
def main(mdl,dataset):
    
    model_path = glob.glob(path_models+'/*'+mdl+'*')[0]
    train_path = glob.glob(path_datasets+'/*'+dataset+'*/*rain*')[0] 
    test_path  = glob.glob(path_datasets+'/*'+dataset+'*/*es*')[0]
    print(model_path,train_path,test_path)


    # In[13]:


    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name'])
    model = AutoModelForCausalLM.from_pretrained(model_path)
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


    # In[ ]:


    hate, cntr_refs = hate_refrences(data,test_set) 
    cntr_replies    = beam_search(hate,model,tokenizer)


    # In[ ]:


    params = [cntr_replies,cntr_refs]  # generated hypothesis, refrences
    bleu, bleu_4, meteor_ = nltk_metrics(params)
    train_corpus = training_corpus(train_set)
    diversity , novelty   = diversity_and_novelty(train_corpus,cntr_replies)


    # In[ ]:


    print(bleu,bleu_4,diversity,novelty,meteor_)


    # In[ ]:


    data_dict ={'hate':hate,'cntr':cntr_replies,'bleu':bleu,'bleu_4':bleu_4,'diversity':diversity,'novelty':novelty, 'meteor':meteor_}
    json.dump(data_dict, open(args["Save_path"] + mdl+"_on_"+dataset+"_updated_"+".json",'w'))

if __name__ == "__main__":
    
    
    prompt = "When she rejected his advance, he grabbed"
    mdl = "Reddit"
    model_path = glob.glob(path_models+'/*'+mdl+'*')[0] 
    model = gen_parallel(base_model = model_path)
    print(model.generate(prompt = prompt))
    exit(0)
    
    mdl     = ["CONAN_DialoGPT", "Create", "Reddit", "Create", "Gab"]
    dataset = ["CONAN", "Reddit", "Reddit", "Gab", "Gab"]
    for i in range(len(mdl)):
        main(mdl[i],dataset[i])
    