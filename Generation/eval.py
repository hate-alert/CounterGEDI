import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import torch
from tqdm import tqdm

import nltk
from nltk.translate import meteor
from nltk.translate.bleu_score import SmoothingFunction


def evaluate(params, model, test_dataloader, device):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
        inputs, labels = (batch[0], batch[0])
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


###################################### BLEU_SCORE , METEOR #######################################
def nltk_metrics(params):
    hypo = params[0]  # a list of generated_hypothesis   
    refs = params[1]  # a list of refrences for particular_refrences    
    
    bleu = bleu_4 = meteor_ = 0.0
    
    for step in range(len(hypo)):
        ref = refs[step]
        hyp = hypo[step]
        bleu    += nltk.translate.bleu_score.sentence_bleu(ref,hyp)
        try:
            bleu_4  += nltk.translate.bleu_score.sentence_bleu(ref,hyp,smoothing_function=SmoothingFunction().method4)
        except:
            pass
        meteor_ += meteor(ref, hyp)
    bleu    /= len(hypo)
    bleu_4  /= len(hypo)
    meteor_ /= len(hypo)
    
    return bleu,bleu_4,meteor_



############################################ JACCARD SIMILARITY #################################
def get_jaccard_sim(str1, str2):   
    if isinstance(str1, float) or isinstance(str2, float):
        return (-1)
    try:
        a = set(str1.split()) 
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        print((str1))
        print(type(str2))
        return 0


############################################### NOVELTY #########################################
def get_novelty(sent,training_corpus):
    max_overlap = 0
    for instance in training_corpus:
        max_overlap = max(max_overlap,get_jaccard_sim(instance,sent))
    return 1-max_overlap

def avg_novelty(sentences,training_corpus):
    avg = 0
    for sent in sentences:
        avg += get_novelty(sent,training_corpus)
    avg = (avg/float(len(sentences)))
    return avg



############################################### DIVERSITY ########################################
def get_diversity(sentences):
    avg = 0.0
    for i in range(len(sentences)):
        max_overlap = 0
        for j in range(len(sentences)):
            if i!=j:
                max_overlap = max(max_overlap,get_jaccard_sim(sentences[i],sentences[j]))
        avg = avg + (1-max_overlap)
    avg = (avg/len(sentences))
    return avg
    
def diversity_and_novelty(training_corpus,gen_replies):
    diversity = get_diversity(gen_replies)
    novelty   = avg_novelty(gen_replies,training_corpus)
    return diversity,novelty
