import torch 
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel


class Model_Label(BertPreTrainedModel):
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels=params['num_classes']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, mask=None, attn=None, labels=None):
        outputs = self.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        
        pooled_output = outputs[1]
        y_pred = self.classifier(self.dropout(pooled_output))
        
        loss_label = None
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            
            
        if(loss_label is not None):
            return y_pred, loss_label
        else:
            return y_pred
        
#### Add Roberta model as well

