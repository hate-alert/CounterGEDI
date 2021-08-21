import numpy as np
from Discriminator.models import *
from Discriminator.data import *
from Discriminator.utils import *
from Discriminator.eval import *
from Generation.models import *
from Generation.utils import *

from Utils.misc import *
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
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

HULK_path='../HULK_new/'




def get_gpu(gpu_id):
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 1.0, maxMemory = 0.7, includeNan=False, excludeID=[], excludeUUID=[])
        print(tempID)
        if len(tempID) > 0 and (tempID[0]==gpu_id):
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)


            

def train(training_dataloader, validation_dataloader, test_dataloader, model, tokenizer, params,run,dict_map,device):
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
    
    
    model.zero_grad()
    optimizer.zero_grad()
    global_step = 0         
    best_macro_f1_val = 0
    best_macro_f1_val = 0
    best_accuracy_val = 0
    best_pre_val = 0
    best_rec_val = 0
    best_gen_val_loss=0
    best_model = None
    # current_epoch, best_weighted_f1 = load_metrics(filepath, model, optimizer
    pt_id = tokenizer.encode('true')[0]
    nt_id = tokenizer.encode('false')[0]
    
    print(pt_id,nt_id)
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch_i in tqdm(range(0, epochs)):
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            b_input_ids=batch[0].to(device).long() 
            b_input_mask=batch[1].to(device)
            b_labels = batch[2].to(device).long()
            
            
            
            batch_0=b_input_ids
            seq_a = (torch.ones(batch_0.shape[0])*pt_id).type_as(batch_0).view(-1,1)
            seq_b = (torch.ones(batch_0.shape[0])*nt_id).type_as(batch_0).view(-1,1)
                

            seq_a = torch.cat((seq_a, batch_0), dim=1)[:,:-1]
            seq_b = torch.cat((seq_b, batch_0), dim=1)[:,:-1]

            bsz = seq_a.shape[0]
            seq_batched = torch.cat((seq_a,seq_b),dim=0)
                #want to compute LM loss here so feeding inputs as labels
            inputs = {"input_ids": seq_batched, "attention_mask": None, "labels": seq_batched}
            #print(seq_batched.shape)
            
            outputs = model(**inputs)
            #print(outputs.loss)
            losses = outputs.loss.view(seq_batched.shape[0], -1)
            #print(losses.shape)
            
            
            
            loss_mask = b_input_mask[:,:-1].to(torch.float32)
#             print(loss_mask.shape)    
            left_ = torch.ones(loss_mask.shape[0],1).type_as(loss_mask)
            loss_mask = torch.cat((left_, loss_mask[:,:-1]), dim=1).to(device)
#             print(loss_mask.shape)    
            loss_lengths = torch.sum(loss_mask,1,keepdim=True)
#             print(loss_lengths)
            
            loss_a,loss_b=torch.split(losses, bsz, dim=0)
            loss_a*=loss_mask
            loss_b*=loss_mask
            gen_loss_a = (b_labels==0).to(torch.float32).unsqueeze(1)*loss_a/loss_lengths
            gen_loss_b = (b_labels==1).to(torch.float32).unsqueeze(1)*loss_b/loss_lengths
            gen_loss = torch.sum(gen_loss_a+gen_loss_b)/bsz
            
                        

            loss_a = (loss_a/loss_lengths).sum(dim=1)
            loss_b= (loss_b/loss_lengths).sum(dim=1)
            
                       
            class_logits = torch.stack((-loss_a, -loss_b), dim=1) #(bsz, 2) dimensional
            b_labels[b_labels == 2] = 1  #turning 3-ary to binary
            class_labels = b_labels
            
            #print(class_logits.shape)
            
            if params['logit_scale']:
                class_logits*=model.logit_scale
                
            if params['bias_own']>0:
                class_logits+=model.bias
            
            loss_fn = torch.nn.CrossEntropyLoss()
            
            
            loss = loss_fn(class_logits, class_labels)
            
            tmp_eval_loss = loss
            tmp_gen_loss = gen_loss
            logits = class_logits
            
            
            loss = loss_fn(class_logits, class_labels)*params['disc_weight'] + params['gen_weight']*gen_loss
            
            
            if params['gradient_accumulation_steps'] > 1:
                loss = loss / params['gradient_accumulation_steps']

            
            if(params['logging']=='local'):
                if step%1000 == 0:
                    print(loss)
            else:
                run["train/batch_loss"].log(loss.item())
            
            
            loss.backward()
            
            if (step + 1) % params['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

            
#        if((global_step+1)%params['save_step']==0):
#                 print("running training")
#                 macro_f1_train,accuracy_train, pre_train, rec_train,overall_gen_loss_train = evaluate_gedi(training_dataloader, params,model,tokenizer,device)

        print("running validation")
        macro_f1_val,accuracy_val, pre_val, rec_val,overall_gen_loss_val = evaluate_gedi(validation_dataloader, params,model,tokenizer,device)


        if(params['logging']=='neptune'):
            #### val scores updated
            run["label/val/f1"].log(macro_f1_val)
            run["label/val/accuracy"].log(accuracy_val)
            run["label/val/positive_class_precision"].log(pre_val)
            run["label/val/positive_class_recall"].log(rec_val)
            run["label/val/gen_loss"].log(overall_gen_loss_val)
        else:
            #print("Train Macro F1: {0:.3f}".format(macro_f1_train))
            print("Val Macro F1: {0:.3f}".format(macro_f1_val))
            print("Val Gen loss: {0:.3f}".format(overall_gen_loss_val))

        if (macro_f1_val > best_macro_f1_val) or (best_gen_val_loss > overall_gen_loss_val):
            best_macro_f1_val = macro_f1_val
            best_accuracy_val = accuracy_val
            best_pre_val = pre_val
            best_rec_val = rec_val
            best_gen_val_loss=overall_gen_loss_val
            save_generation_gedi(model,tokenizer,params)

                    

#         print("running test")
#         macro_f1_test,accuracy_test, pre_test, rec_test,overall_gen_loss_test = evaluate_gedi(test_dataloader, params,model,tokenizer,device)
#         if(params['logging']=='neptune'):
#             #### test scores updated
#             run["label/test/f1"].log(macro_f1_test)
#             run["label/test/accuracy"].log(accuracy_test)
#             run["label/test/positive_class_precision"].log(pre_test)
#             run["label/test/positive_class_recall"].log(rec_test)
#             run["label/test/gen_loss"].log(overall_gen_loss_test)

#         else:
#             print("Test Macro F1: {0:.3f}".format(macro_f1_test))
#             print("Test Gen loss: {0:.3f}".format(overall_gen_loss_test))

                
    if(params['logging']=='neptune'):
      
        run["label/val/best_f1"].log(best_macro_f1_val)
        run["label/val/best_accuracy"].log(best_accuracy_val)
        run["label/val/best_positive_class_precision"].log(best_pre_val)
        run["label/val/best_positive_class_recall"].log(best_rec_val)        


def train_caller(params,run=None,gpu_id=0):
        tokenizer = AutoTokenizer.from_pretrained(params['model_path'],use_fast=False, cache_dir=params['cache_path'])
        ### add model loading code 
        tokenizer.pad_token = '[PAD]'
        dataset_path=HULK_path+'Counterspeech/Datasets/'+params['task_name']+'/'
        train_data,valid_data,test_data,class_label=load_data_own(data_path=dataset_path)
        print(class_label)
        params['num_classes']=len(class_label)
        dict_map=None
        if(params['label_positive'] in class_label):
            dict_map={}
            for label in class_label:
                if(label==params['label_positive']):
                    dict_map[label]='true'
                else:
                    dict_map[label]='false'
        else:
            print("labels should be one of",class_label)
        print(dict_map)
        
        
        ## set discriminator loss
        params['disc_weight']=1-params['gen_weight']
        train_data_source = Normal_Dataset(train_data,class_label,dict_map,tokenizer, params,train = True)
        val_data_source = Normal_Dataset(valid_data,class_label,dict_map,tokenizer,params)
        test_data_source = Normal_Dataset(test_data,class_label,dict_map,tokenizer, params)
        
        
    
        
        if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            ##### You can set the device manually if you have only one gpu
            ##### comment this line if you don't want to manually set the gpu
            deviceID = get_gpu(gpu_id)
            torch.cuda.set_device(deviceID[0])
            #### comment this line if you want to manually set the gpu
            #### required parameter is the gpu id
            #torch.cuda.set_device(args.gpuid)

        else:
            print('Since you dont want to use GPU, using the CPU instead.')
            device = torch.device("cpu")
        config = AutoConfig.from_pretrained(params['model_path'],cache_dir=params['cache_path'])
        config.reduction='None'
        
        if(params['bias_own']>0):
            config.bias=params['bias_own']
        config.logit_scale=params['logit_scale']
        print(params['bias_own'])
        print(params['logit_scale'])
        model = Model_Generation.from_pretrained(params['model_path'],config=config,cache_dir=params['cache_path']).to(device)
        
        
        for param in model.transformer.wpe.parameters():
                param.requires_grad = False
        for param in model.transformer.wte.parameters():
                param.requires_grad = False

        train(train_data_source.DataLoader, val_data_source.DataLoader,test_data_source.DataLoader,model,tokenizer,params,run,dict_map,device)
        
    
params={
 'model_path':'gpt2',
 'task_name':'Emotion',
 'save_path':HULK_path+'Counterspeech/Saved_models/Discriminator/',
 'logging':'neptune',
 'cache_path':HULK_path+'Saved_models/',
 'label_positive':'love',
 'batch_size':8,
 'max_length':128,
 'dropout':1.0,
 'device':'cuda',
 'epochs':5,
 'seed':42,
 'learning_rate':2e-5,
 'bias_own':2,
 'logit_scale':True,
 'gradient_accumulation_steps':1,
 'gen_weight':0.8,
 'max_grad_norm':1,   
 'save_step':1000

}
if __name__ == "__main__":
    fix_the_random(seed_val = params['seed'])
    
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('task',
                           metavar='--task',
                           type=str,
                           help='the task of the model')
    
    my_parser.add_argument('label',
                           metavar='--label',
                           type=str,
                           help='the label corresponding to data')
    my_parser.add_argument('gpu_id',
                           metavar='--gpu_id',
                           type=int,
                           help='GPU id')
    
    
    
    args = my_parser.parse_args()
    
    params['task_name']=args.task
    params['label_positive']=args.label
    
    run=None
    if(params['logging']=='neptune'):
        run = neptune.init(project=project_name,api_token=api_token)
        run["parameters"] = params
    else:
        pass
    train_caller(params,run,args.gpu_id)
    if(run is not None):
        run.stop()
    
