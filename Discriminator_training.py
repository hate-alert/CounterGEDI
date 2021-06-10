import numpy as np
from Discriminator.models import *
from Discriminator.data import *
from Discriminator.utils import *
from Discriminator.eval import *

from Utils.misc import *
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
import neptune.new as neptune
import GPUtil
import numpy as np
#from datasets import list_datasets, load_dataset
from apiconfig import *
import pandas as pd
from tqdm import tqdm
import argparse
import json
import time

params={
 'dataset':'../HULK/Counterspeech/Datasets/Politeness/',
 'model_path':'gpt2-medium',
 'task_name':'Politeness',
 'save_path':'../HULK/Counterspeech/Saved_models/Discriminator/',
 'logging':'local',
 'cache_path':'../HULK/Saved_models/',
 'batch_size':16,
 'max_length':128,
 'dropout':1.0,
 'device':'cpu',
 'epochs':10,
 'seed':42,
 'learning_rate':1e-5

}


def get_gpu():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.9, maxMemory = 0.7, includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)


            

def train(training_dataloader, validation_dataloader, test_dataloader, model, tokenizer, params,run,device):
    epochs=params['epochs']
    total_steps = len(training_dataloader) * epochs
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['learning_rate'], eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    best_macro_f1_val = 0
    best_macro_f1_test = 0
    best_accuracy_test = 0
    best_pre_test = 0
    best_rec_test = 0
    
    best_model = None
    # current_epoch, best_weighted_f1 = load_metrics(filepath, model, optimizer)

    criterion = nn.CrossEntropyLoss()

    for epoch_i in tqdm(range(0, epochs)):
        model.train()
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            b_input_ids=batch[0].to(device).long() 
#             b_input_mask=batch[1].to(device)
            b_labels = batch[1].to(device).long()
            ypred, loss = model(b_input_ids,b_labels)
            
            if(params['logging']=='local'):
                if step%100 == 0:
                    print(loss.item())
            else:
                run["train/batch_loss"].log(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        
        print("running validation")
        macro_f1_val,accuracy_val, pre_val, rec_val = evaluate_classifier(validation_dataloader, params,model,device)
        
        print("running test")
        macro_f1_test,accuracy_test, pre_test, rec_test = evaluate_classifier(test_dataloader, params,model,device)
       
        if(params['logging']=='neptune'):
            #### val scores updated
            run["label/val/f1"].log(macro_f1_val)
            run["label/val/accuracy"].log(accuracy_val)
            run["label/val/positive_class_precision"].log(pre_val)
            run["label/val/positive_class_recall"].log(rec_val)
            
            #### test scores updated
            run["label/test/f1"].log(macro_f1_test)
            run["label/test/accuracy"].log(accuracy_test)
            run["label/test/positive_class_precision"].log(pre_test)
            run["label/test/positive_class_recall"].log(rec_test)
            
        else:
            print("  Macro F1: {0:.3f}".format(macro_f1_val))
        
        if macro_f1_val > best_macro_f1_val:
            best_macro_f1_val = macro_f1_val
            best_macro_f1_test = macro_f1_test
            best_accuracy_test = accuracy_test
            best_pre_test = pre_test
            best_rec_test = rec_test
            save_detection_model(model,tokenizer,params)
            
    if(params['logging']=='neptune'):
      
        run["label/test/best_f1"].log(best_macro_f1_test)
        run["label/test/best_accuracy"].log(best_accuracy_test)
        run["label/test/best_positive_class_precision"].log(best_pre_test)
        run["label/test/best_positive_class_recall"].log(best_rec_test)        


def train_caller(params,run=None):
        tokenizer = AutoTokenizer.from_pretrained(params['model_path'],use_fast=False, cache_dir=params['cache_path'],local_files_only=False)
        ### add model loading code 
        tokenizer.pad_token = tokenizer.eos_token

        
        dataset_path='../HULK/Counterspeech/Datasets/'+params['task_name']+'/'
        train_data,valid_data,test_data,class_label=load_data_own(data_path=dataset_path)
        print(class_label)
        params['num_classes']=len(class_label)
        train_data_source = Normal_Dataset(train_data,class_label,tokenizer, params,train = True)
        val_data_source = Normal_Dataset(valid_data,class_label,tokenizer,params)
        test_data_source = Normal_Dataset(test_data,class_label,tokenizer, params)
        
        if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
    #         deviceID = get_gpu()
    #         torch.cuda.set_device(deviceID[0])
            ##### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            torch.cuda.set_device(args.gpuid)

        else:
            print('Since you dont want to use GPU, using the CPU instead.')
            device = torch.device("cpu")
        
        model = Discriminator(params,tokenizer=tokenizer,device=device).to(device)
        
#         model = Model_Label.from_pretrained(params['model_path'], cache_dir=params['cache_path'],params=params,output_attentions = True,output_hidden_states = False).to(device)
        train(train_data_source.DataLoader, val_data_source.DataLoader,test_data_source.DataLoader,model,tokenizer,params,run,device)
        
    
    
if __name__ == "__main__":
    fix_the_random(seed_val = params['seed'])
    train_caller(params)
    
