from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import re


class Hatexplain_dataset():
    def __init__(self, data, params=None, tokenizer=None, train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_len = self.params['max_length']
        if params['num_classes'] == 3:
            self.label_dict = {0: 0,
                                1: 1,
                                2: 2}
        elif params['num_classes'] == 2:
            self.label_dict = {0: 1,
                                1: 0,
                                2: 1}
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.labels, self.attn = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels)
        
    def tokenize(self, sentences):
        input_ids, attention_masks, token_type_ids = [], [], []
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return {'input_ids': input_ids, 'attention_masks': attention_masks}
    
    
    
    
    def process_data(self, data):
        sentences, labels, attn = [], [], []
        print(len(data))
        for row in data:
            word_tokens_all, word_mask_all = returnMask(row, self.tokenizer)
            label = max(set(row['annotators']['label']), key = row['annotators']['label'].count)
            if(self.params['train_att']):
                at_mask = self.process_mask_attn(word_mask_all,label)
            elif(self.params['train_rationale']):
                at_mask = self.process_masks(word_mask_all)
            else:
                at_mask = []
            sentence = ' '.join(row['post_tokens'])
            sentences.append(sentence)
            labels.append(self.label_dict[label])
            at_mask = at_mask + [0]*(self.max_len-len(at_mask))
            attn.append(at_mask)
        inputs = self.tokenize(sentences)
        return inputs, torch.Tensor(labels), torch.Tensor(attn)
    
    def get_dataloader(self, inputs, attn, labels, train = True):
        data = TensorDataset(inputs['input_ids'], inputs['attention_masks'], attn, labels)
        if self.train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.batch_size, drop_last=True)
    
    
    
    
class Normal_Dataset():
    def __init__(self, data, class_labels=None, params=None, tokenizer=None, train = False):
        self.data = data
        self.params= params
        self.batch_size = self.params['batch_size']
        self.train = train
        self.max_len = self.params['max_length']
        
        
        self.label_dict={}
        count_label=0
        for label_name in class_labels:
            self.label_dict[label_name]=count
        self.max_length=params['max_length']        
        self.count_dic = {}
        self.tokenizer = tokenizer
        self.inputs, self.labels, self.attn = self.process_data(self.data)
        self.DataLoader = self.get_dataloader(self.inputs, self.attn, self.labels)
    
    def preprocess_func(self, text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent
    
#     def dummy_attention(self,inputs):
#         attn=[]
#         for sent in inputs['input_ids']:
#             temp=[0]*len(sent)
#             attn.append(temp)
#         return attn
    
    def process_data(self, data):
        sentences, labels, attn = [], [], []
        for label, sentence in zip(list(data['label']), list(data['text'])):
            label = self.label_dict[label]
            try:
                sentence = self.preprocess_func(sentence)
            except TypeError:
                sentence = self.preprocess_func("dummy text")
            sentences.append(sentence)
            labels.append(label)
        inputs = self.tokenize(sentences)
        #attn = self.dummy_attention(inputs)
        #return inputs, torch.Tensor(labels), torch.Tensor(attn)
        return inputs, torch.Tensor(labels)