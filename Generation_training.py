import transformers 
import os
import json
import torch
import time
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple
from keras.preprocessing.sequence import pad_sequences
from tqdm import trange
from tqdm import tqdm
import numpy as np
import GPUtil
from Utils import misc,preprocess
from sklearn.model_selection import train_test_split
from apiconfig import project_name,api_token
import neptune.new as neptune
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
import numpy as np
from Generation.models import *
from Generation.data import *
from Generation.eval import *
from Generation.utils import *

from Utils.misc import *

model_memory=9
total_memory=16

# Check for availability of GPU and accordingly return the GPU id available to use
def get_gpu(gpu_id):
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 2, maxLoad = 1.0, maxMemory = (1-(model_memory/total_memory)), includeNan=False, excludeID=[], excludeUUID=[])
        for i in range(len(tempID)):
            if len(tempID) > 0 and (tempID[i]==gpu_id):
                print("Found a gpu")
                print('We will use the GPU:',tempID[i],torch.cuda.get_device_name(tempID[i]))
                deviceID=[tempID[i]]
                return deviceID
            else:
                time.sleep(5)
                

def train(params,train_dataloader, eval_dataloader, test_dataloader, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device,run):
    """ Train the model """
    
    if params['max_steps'] > 0:
        t_total = params['max_steps']
        params['num_train_epochs'] = params['max_steps'] // (len(train_dataloader) // params['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // params['gradient_accumulation_steps'] * params['num_train_epochs']
    
    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    print("Tokenizer loaded")
    # add_special_tokens_(model, tokenizer)
    # Prepare optimizer and schedule (linear warmup and decay)
    # Track metadata and hyperparameters of your run
    # Track the training process by logging your training metrics
    
    #The optimizer allows us to apply different hyperpameters for specific parameter groups. 
    #For example, we can apply weight decay to all parameters other than bias and layer normalization terms:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": params['weight_decay'],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['learning_rate'], eps=params['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=params['warmup_steps'], num_training_steps=t_total
    )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(params['num_train_epochs']), desc="Epoch")
    eval_best_val = 100000
    eval_best_test = 100000
    eval_val = []
    eval_test = []
    epoch_count=1
    for _ in train_iterator:
        print("Current running epoch", epoch_count)
        epoch_count+=1
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch[0], batch[0])
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

#             if params['n_gpu'] > 1:
#                 loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if params['gradient_accumulation_steps'] > 1:
                loss = loss / params['gradient_accumulation_steps']

            loss.backward()

            tr_loss += loss.item()
            if(params['logging']=='neptune'):
                run["train/batch_loss"].log(loss.item())
            
            if((step+1)% 1000==0):
                print('Average batch loss',  tr_loss/(step+1))
               
               
            if (step + 1) % params['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
          
            if params['max_steps'] > 0 and global_step > params['max_steps']:
                epoch_iterator.close()
                break
        eval_train_score=evaluate(params, model, train_dataloader, device)
#         #eval_train_score=evaluate(params, model, tokenizer, df_trn,device,params['block_size'])
        eval_val_score=evaluate(params, model, eval_dataloader, device)
        eval_test_score=evaluate(params, model, test_dataloader, device)

        
        if(params['logging']=='neptune'):
            run["eval/perplexity_train"].log(eval_train_score)
            run["eval/perplexity_val"].log(eval_val_score)
            run["eval/perplexity_test"].log(eval_test_score)
        else:
            print("perplexity train score", eval_train_score)
            print("perplexity val score", eval_val_score)
            print("perplexity test score", eval_test_score)

           
        eval_val.append(eval_val_score)
        eval_test.append(eval_test_score)
        if params['max_steps'] > 0 and global_step > params['max_steps']:
            train_iterator.close()
            break
        if eval_val[-1] < eval_best_val:
            save_generation_model(model,tokenizer, params)
            eval_best_val = eval_val[-1]
            eval_best_test = eval_test[-1]
    if(params['logging']=='neptune'):
        run["eval/best_perplexity_val"]=eval_best_val
        run["eval/best_perplexity_test"]=eval_best_test
    else:
        print("best perplexity val", eval_best_val)
        print("best perplexity test", eval_best_test)
    
    return global_step, tr_loss / global_step, eval_val

def train_caller(params,run=None):
    
    dataset_path='../HULK/Counterspeech/Datasets/'+params['task_name']+'/'       # Path to the datatset defined as "All_Dataset_folder/" + params["task_name"] + "/"
    config = AutoConfig.from_pretrained(params['model_path'],cache_dir=params['cache_path'])
    tokenizer = AutoTokenizer.from_pretrained(params['model_path'],cache_dir=params['cache_path'],fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    train_data,valid_data,test_data=load_data_own_gen(data_path=dataset_path)	 # Load the train, validation and test data from Dataset_paths
										 # Assume data_path contains three files Train.csv, Val.csv and Test.csv 
    train_data_source = Normal_Generation_Dataset(train_data,tokenizer, params,train = True, topic=params['topic'])		# Preprocess the dataset 
    val_data_source = Normal_Generation_Dataset(valid_data,tokenizer,params, topic=params['topic'])
    test_data_source = Normal_Generation_Dataset(test_data,tokenizer, params, topic=params['topic'])
    
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
            deviceID = get_gpu(1)
            torch.cuda.set_device(deviceID[0])
            ##### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            #torch.cuda.set_device(0)

    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    model = Model_Generation.from_pretrained(params['model_path'],config=config,cache_dir=params['cache_path'])
    for param in model.transformer.wpe.parameters():
            param.requires_grad = False
    for param in model.transformer.wte.parameters():
            param.requires_grad = False


    if params['freeze_layer_count'] != -1:
         # otherwise we freeze the first `freeze_layer_count` encoder layers
        for layer in model.transformer.h[:params['freeze_layer_count']]:
            for param in layer.parameters():
                param.requires_grad = False
    model.resize_token_embeddings(len(tokenizer))
    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    # Training
    train(params,train_data_source.DataLoader, val_data_source.DataLoader, test_data_source.DataLoader, model, tokenizer, device,run)




params={
     'save_path':'../HULK/Counterspeech/Saved_models/Generator/',
     'model_path':'microsoft/DialoGPT-small',
     'cache_path':'../HULK/Saved_models/',
     'task_name':'Create_debate_summarised',
     'topic': True,
     'max_length': 512,
     'train': True,
     'batch_size':4,
     'gradient_accumulation_steps':1,
     'learning_rate':5e-6,
     'weight_decay':0.0,
     'adam_epsilon':1e-8,
     'max_grad_norm':1.0,
     'num_train_epochs':10,
     'max_steps':-1,
     'warmup_steps':0,
     'seed':42,
     'device':'cuda',
     'logging':'local',
     'freeze_layer_count':0
}



if __name__ == "__main__":
    fix_the_random(seed_val = params['seed'])
    run=None
    if(params['logging']=='neptune'):
        run = neptune.init(project=project_name,api_token=api_token)
        run["parameters"] = params
    else:
        pass
    train_caller(params,run)
    if(run is not None):
        run.stop()
    
    
    
    