import torch
import numpy as np
import pandas as pd
import os 
import random
def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def load_data_own(data_path='Davidson'):
    '''data_path: The folder path of the dataset in csv
    '''
    #path_own=os.path.dirname(os.path.realpath(__file__))
    #print(path_own)
    temp_path_train=data_path+'Train.csv'
    df_train=pd.read_csv(temp_path_train)
    temp_path_val=data_path+'Val.csv'
    df_val=pd.read_csv(temp_path_val)
    temp_path_test=data_path+'Test.csv'
    df_test=pd.read_csv(temp_path_test)
    labels=sorted(list(df_train['labels'].unique()))
    df_train=df_train.sample(1000)
    df_val=df_val.sample(1000)
    df_test=df_test.sample(1000)
    return df_train,df_val,df_test,labels



def load_data_own_gen(data_path='Davidson'):
    '''data_path: The folder path of the dataset in csv
    '''
    #path_own=os.path.dirname(os.path.realpath(__file__))
    #print(path_own)
    temp_path_train=data_path+'Train.csv'
    df_train=pd.read_csv(temp_path_train)
    temp_path_val=data_path+'Val.csv'
    df_val=pd.read_csv(temp_path_val)
    temp_path_test=data_path+'Test.csv'
    df_test=pd.read_csv(temp_path_test)
    return df_train,df_val,df_test