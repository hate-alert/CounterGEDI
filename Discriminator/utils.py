import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function
import os 
import random
import numpy as np

def save_detection_model(model,tokenizer,params):
    output_dir = params['save_path']+params['task_name']+'_'+params['model_path']+'/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
