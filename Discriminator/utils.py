import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function
import os 
import random
import numpy as np
import json

def save_detection_model(discriminator,tokenizer,params):
    output_dir = params['save_path']+params['task_name']+'_'+params['model_path']+'/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    torch.save(discriminator.get_classifier().state_dict(),
                       output_dir+'model.pt')

#     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
#     model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    
    
def save_detection_meta(discriminator_meta,params):
    output_dir = params['save_path']+params['task_name']+'_'+params['model_path']+'/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_dir+'disctiminator.json', "w") as meta_file:
            json.dump(discriminator_meta, meta_file)
