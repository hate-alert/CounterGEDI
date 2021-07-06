from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import re
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons])




class Normal_Generation_Dataset():
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs)
    
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        
        
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
    
    def construct_conv(self,dict_reply_pair):
        conv = None
        flatten = lambda l: [item for sublist in l for item in sublist]
        initiator=self.preprocess_func(dict_reply_pair['initiator_message'])
        reply=self.preprocess_func(dict_reply_pair['reply_message'])


        conv = list([self.tokenizer.encode(initiator,truncation=True,max_length=int((self.max_length/2)-1))+ 
                     [self.tokenizer.eos_token_id] + 
                    self.tokenizer.encode(reply,truncation=True,max_length=int((self.max_length/2)-1))+
                    [self.tokenizer.eos_token_id]])

        conv = flatten(conv)
        return conv

    def tokenize(self, dataframe):
        inputs=[]
        for index,row in tqdm(dataframe.iterrows(),total=len(dataframe)):
            conv = self.construct_conv(row)
            inputs.append(conv)
        return inputs
    
    def process_data(self, data):
        inputs = self.tokenize(data)
        return inputs
    
    def get_dataloader(self, inputs):
        inputs = pad_sequences(inputs,maxlen=int(self.params['max_length']), dtype="long", 
                          value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        inputs = torch.tensor(inputs)
        data = TensorDataset(inputs)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size, drop_last=True)


    
    
    
    
    
    
    
class Normal_Dexpert_Dataset(Normal_Generation_Dataset):
    def __init__(self, data, tokenizer=None,  params=None,train = False):
        self.params= params
        self.label = params['label']
        data=data[data['label']==self.label]
        self.data = data
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs)
    
    
    
    def construct_conv(self,dict_reply_pair):
        conv = None
        flatten = lambda l: [item for sublist in l for item in sublist]
        initiator=self.preprocess_func(dict_reply_pair['text'])
        conv = list(self.tokenizer.encode(initiator,truncation=True,max_length=int(self.max_length)))

        conv = flatten(conv)
        return conv