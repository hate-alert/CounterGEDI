import transformers 
import os
import json
import torch
import time
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from tqdm import tqdm
import numpy as np
import neptune.new as neptune
import GPUtil
from Utils import misc,preprocess
from sklearn.model_selection import train_test_split
from apiconfig import project_name,api_token
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







def load_and_cache_examples(args, tokenizer, text,block_size):
    return preprocess.ConversationDataset(tokenizer, args, text,block_size)




def get_gpu():
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while(1):
        tempID = [] 
        tempID = GPUtil.getAvailable(order = 'memory', limit = 1, maxLoad = 0.1, maxMemory = 0.07, includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:',tempID[0],torch.cuda.get_device_name(tempID[0]))
            deviceID=tempID
            return deviceID
        else:
            time.sleep(5)

def train(params, train_dataset, eval_dataset,model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device,run):
    """ Train the model """
    
    params['train_batch_size'] = params['per_gpu_train_batch_size'] * max(1, params['n_gpu'])

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    train_dataset = load_and_cache_examples(params, tokenizer, train_dataset,params['block_size'])
    train_sampler = RandomSampler(train_dataset) if params['local_rank'] == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=params['train_batch_size'], collate_fn=collate, drop_last = True
    )

    if params['max_steps'] > 0:
        t_total = params['max_steps']
        params['num_train_epochs'] = params['max_steps'] // (len(train_dataloader) // params['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // params['gradient_accumulation_steps'] * params['num_train_epochs']

    model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model.resize_token_embeddings(len(tokenizer))
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
    train_iterator = trange(
        epochs_trained, int(params['num_train_epochs']), desc="Epoch", disable=params['local_rank'] not in [-1, 0]
    )
    eval_best = 100000
    eval_ = []
    epoch_count=1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=params['local_rank'] not in [-1, 0])
        print("Current running epoch", epoch_count)
        epoch_count+=1
        epoch_count+=1
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch, batch)
            if inputs.shape[1] > 1024: continue
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if params['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if params['gradient_accumulation_steps'] > 1:
                loss = loss / params['gradient_accumulation_steps']

            loss.backward()

            tr_loss += loss.item()
            if(params['logging']=='neptune'):
                run["train/batch_loss"].log(loss.item())
            
            if((step+1)% params['logging_steps']==0):
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
        eval_train_score=evaluate(params, model, tokenizer, train_dataset,device,params['block_size'])
        eval_score=evaluate(params, model, tokenizer, eval_dataset,device,params['block_size'])
        if(params['logging']=='neptune'):
            run["eval/perplexity_train"]=eval_train_score
            run["eval/perplexity_val"]=eval_score
        else:
            print("perplexity train score", eval_train_score)
            print("perplexity val score", eval_score)
           
        eval_.append(eval_score)
        if params['max_steps'] > 0 and global_step > params['max_steps']:
            train_iterator.close()
            break
        if eval_[-1] < eval_best:
            os.makedirs(params['output_dir'], exist_ok=True)

            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(params['output_dir'])
            tokenizer.save_pretrained(params['output_dir'])

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(params['output_dir'], "training_args.bin"))
            eval_best = eval_[-1]
    
    if(params['logging']=='neptune'):
        run["eval/best_perplexity_val"]=eval_best
    else:
        print("best perplexity val", eval_best)
    
    return global_step, tr_loss / global_step, eval_






# Evaluation of some model
def evaluate(params, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, df_val,device,block_size):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = params['output_dir']

    eval_dataset = load_and_cache_examples(params, tokenizer, df_val,  block_size)
    os.makedirs(eval_output_dir, exist_ok=True)
    params['eval_batch_size'] = params['per_gpu_eval_batch_size'] * max(1,params['n_gpu'])
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=params['eval_batch_size'], collate_fn=collate, drop_last = True
    )
    
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    return perplexity






params={
     'output_dir':'../HULK/Counterspeech/models/createdebate_model',
     'model_type':'microsoft/DialoGPT-medium',
     'model_name_or_path':'microsoft/DialoGPT-medium',
     'config_name':'microsoft/DialoGPT-medium',
     'tokenizer_name':'microsoft/DialoGPT-medium',
     'cache_dir':'../HULK/Saved_models/',
     'block_size': 512,
     'do_train': True,
     'do_eval':True,
     'evaluate_during_training':False,
     'per_gpu_train_batch_size':4,
     'per_gpu_eval_batch_size':4,
     'gradient_accumulation_steps':1,
     'learning_rate':5e-6,
     'weight_decay':0.0,
     'adam_epsilon':1e-8,
     'max_grad_norm':1.0,
     'num_train_epochs':10,
     'max_steps':-1,
     'warmup_steps':0,
     'logging_steps':1000,
     'save_steps':3500,
     'save_total_limit':None,
     'eval_all_checkpoints':False,
     'no_cuda':False,
     'overwrite_output_dir':True,
     'overwrite_cache':True,
     'should_continue':False,
     'n_gpu':1,
     'seed':56,
     'local_rank':-1,
     'fp16':False,
     'fp16_opt_level':'O1',
     'device':'cpu',
     'logging':'neptune',
     'freeze_layer_count':6
}



if __name__ == "__main__":
    
    
    with open('../HULK/Counterspeech/selected_arguments.json', 'r') as fp:
        dict_urls=json.load(fp)
    total_data_sentences=[]
    for key in dict_urls:
        total_data_sentences+=dict_urls[key]['selected_arguments']

    X_train, X_test_dev = train_test_split(total_data_sentences, test_size=0.2, random_state=42, shuffle=True)
    X_test, X_dev = train_test_split(X_test_dev, test_size=0.5, random_state=42, shuffle=True)
    
    
    path=params['cache_dir']
#     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small",cache_dir=path)
#     model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small",cache_dir=path)

    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        ##### You can set the device manually if you have only one gpu
        ##### comment this line if you don't want to manually set the gpu
        #deviceID = get_gpu()
        #torch.cuda.set_device(deviceID[0])
        ##### comment this line if you want to manually set the gpu
        #### required parameter is the gpu id
        torch.cuda.set_device(0)
        
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    run=None
    if(params['logging']=='neptune'):
        run = neptune.init(project=project_name,api_token=api_token)
        run["Dataset"] = "createdebate_dataset"
        run["parameters"] = params

    config = AutoConfig.from_pretrained(params['config_name'],cache_dir=path)
    tokenizer = AutoTokenizer.from_pretrained(params['tokenizer_name'],cache_dir=path)
    model = AutoModelForCausalLM.from_pretrained(
        params['model_name_or_path'],
        from_tf=False,
        config=config,
        cache_dir=path
    )
    for param in model.transformer.wpe.parameters():
                param.requires_grad = False
    for param in model.transformer.wte.parameters():
            param.requires_grad = False


    if params['freeze_layer_count'] != -1:
         # otherwise we freeze the first `freeze_layer_count` encoder layers
        for layer in model.transformer.h[:params['freeze_layer_count']]:
            for param in layer.parameters():
                param.requires_grad = False

    model.to(device)

    # Training
    if params['do_train']:
        global_step, tr_loss, eval_ = train(params, X_train, X_dev, model, tokenizer, device,run)
        
    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if params['do_train']:
        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(params['output_dir'])
        tokenizer = AutoTokenizer.from_pretrained(params['output_dir'])
        model.to(device)
    test_eval=0
    if test_text!=None:
        test_eval = evaluate(params, model, tokenizer, test_text)[1]
        if(params['logging']=='neptune'):
            run["eval/best_perplexity_test"]=test_eval
           
        else:
            print("eval/best_perplexity_test",test_eval)
           
    run.stop()
    
    
    
    
    