import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import torch
from tqdm import tqdm

def get_predicted(preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat


def evaluate_classifier(test_dataloader, params,model,device):
    model.eval()
    y_preds, y_test = np.array([]), np.array([])

    total = 0
    correct = 0
    pred = []
    label = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
        with torch.no_grad():         
            ypred, _ = model(b_input_ids, b_input_mask,labels=b_labels)

        ypred = ypred.cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
#         print("ypred")
#         print(get_predicted(ypred))
        
#         print("labels")
#         print(label_ids)
        
        try:
            y_preds = np.hstack((y_preds, get_predicted(ypred)))
            y_test = np.hstack((y_test, label_ids))
        except:
            y_preds, y_test = get_predicted(ypred), label_ids
    
#     print(classification_report(y_test, y_preds))
#     f1 = f1_score(y_test, y_preds, average = 'macro')

    testf1=f1_score(y_test, y_preds, average='macro')
    testacc=accuracy_score(y_test,y_preds)
    testprecision=precision_score(y_test, y_preds, average='binary')
    testrecall=recall_score(y_test, y_preds, average='binary')
    return testf1,testacc,testprecision,testrecall 




def evaluate_gedi(test_dataloader, params,model,tokenizer,device):
    model.eval()
    
    total = 0
    correct = 0
    pred = []
    label = []
    preds = None
    out_label_ids = None

    eval_loss=0
    overall_gen_loss=0
    nb_eval_steps =0
    
    pt_id = tokenizer.encode('true')[0]
    nt_id = tokenizer.encode('false')[0]
    
    print(pt_id,nt_id)
    
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
        
        with torch.no_grad():
            batch_0=b_input_ids
            seq_a = (torch.ones(batch_0.shape[0])*pt_id).type_as(batch_0).view(-1,1)
            seq_b = (torch.ones(batch_0.shape[0])*nt_id).type_as(batch_0).view(-1,1)
                

            seq_a = torch.cat((seq_a, batch_0), dim=1)[:,:-1]
            seq_b = torch.cat((seq_b, batch_0), dim=1)[:,:-1]

            bsz = seq_a.shape[0]
            seq_batched = torch.cat((seq_a,seq_b),dim=0)
                #want to compute LM loss here so feeding inputs as labels
            inputs = {"input_ids": seq_batched, "attention_mask": None, "labels": seq_batched}
            
            outputs = model(**inputs)
            #print(outputs.loss)
            losses = outputs.loss.view(seq_batched.shape[0], -1)
            
            
            
            loss_mask = b_input_mask[:,:-1].to(torch.float32).cuda()
            left_ = torch.ones(loss_mask.shape[0],1).type_as(loss_mask)
            loss_mask = torch.cat((left_, loss_mask[:,:-1]), dim=1).to(device)
            loss_lengths = torch.sum(loss_mask,1,keepdim=True)
            
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
            if params['logit_scale']:
                class_logits*=model.logit_scale
                
            if params['bias_own']>0:
                class_logits+=model.bias
            
            loss_fn = torch.nn.CrossEntropyLoss()
            
            
            loss = loss_fn(class_logits, class_labels)
            
            tmp_eval_loss = loss
            tmp_gen_loss = gen_loss
            logits = class_logits
            
            eval_loss += tmp_eval_loss.mean().item()
            overall_gen_loss += tmp_gen_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = class_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, class_labels.detach().cpu().numpy(), axis=0)
            

    eval_loss = eval_loss / nb_eval_steps
    overall_gen_loss = overall_gen_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    
    print(preds)
    print(out_label_ids)
    
    testf1=f1_score(out_label_ids, preds, average='macro')
    testacc=accuracy_score(out_label_ids,preds)
    testprecision=precision_score(out_label_ids, preds, average='binary')
    testrecall=recall_score(out_label_ids, preds, average='binary')

        
        
    return testf1,testacc,testprecision,testrecall,overall_gen_loss 
