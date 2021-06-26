import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function
import os 
import random
import numpy as np
import json

def save_generation_model(model,tokenizer,params):
    
    if len(params['model_path'].split('/'))>1:
        params['model_path']=params['model_path'].split('/')[1]
    
    output_dir = params['save_path']+params['task_name']+'_'+params['model_path']+'/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(params, os.path.join(output_dir, "training_args.bin"))