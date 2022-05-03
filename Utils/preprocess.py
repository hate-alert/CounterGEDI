from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from torch.utils.data import Dataset
import torch
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

import re

##### text preprocessor for ekphrasis
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


def preprocess_function(text):
    remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
    word_list=text_processor.pre_process_doc(text)
    word_list=list(filter(lambda a: a not in remove_words, word_list)) 
    sent=" ".join(word_list)
    sent = re.sub(r"[<\*>]", " ",sent)
    return sent


def construct_conv(dict_reply_pair, tokenizer, eos = True, block_size=512, dataset="Debate"):
    conv = None
    if dataset=="CONAN":
        flatten = lambda l: [item for sublist in l for item in sublist]
        conv = list([tokenizer.encode(text) + [tokenizer.eos_token_id]])
        conv = flatten(conv)

    elif dataset=="Debate":
        flatten = lambda l: [item for sublist in l for item in sublist]
        initiator=preprocess_function(dict_reply_pair['initiator_message'])
        reply=preprocess_function(dict_reply_pair['reply_message'])


        conv = list([tokenizer.encode(initiator,truncation=True,max_length=int((block_size/2)-1))+ 
                     [tokenizer.eos_token_id] + 
                    tokenizer.encode(reply,truncation=True,max_length=int((block_size/2)-1))+
                    [tokenizer.eos_token_id]])

        conv = flatten(conv)
        
    return conv

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, params, text_list, block_size=512):

        self.examples = []
        for element in text_list:
            conv = construct_conv(element, tokenizer,block_size)
            self.examples.append(conv)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


if __name__ == "__main__":
    params={
     'include_special':False,
    }
    text="There is #nigger ..... loooong"
    print(preprocess_function(text,params))
    