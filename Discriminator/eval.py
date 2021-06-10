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
        try:
            y_preds = np.hstack((y_preds, get_predicted(ypred)))
            y_test = np.hstack((y_test, label_ids))
        except:
            y_preds, y_test = ypred, label_ids
    
#     print(classification_report(y_test, y_preds))
#     f1 = f1_score(y_test, y_preds, average = 'macro')

    testf1=f1_score(y_test, y_preds, average='macro')
    testacc=accuracy_score(y_test,y_preds)
    testprecision=precision_score(y_test, y_preds, average='binary')
    testrecall=recall_score(y_test, y_preds, average='binary')
    return testf1,testacc,testprecision,testrecall 
