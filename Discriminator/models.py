import torch 
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel,GPT2Tokenizer,GPT2Model

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits


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

class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(self, params,classifier_head=None,tokenizer=None,device=None):
#         class_size=None,
#         pretrained_model="gpt2-medium",
#         classifier_head=None,
#         cached_mode=False,
#         device='cpu'
        self.EPSILON = 1e-10
        self.pretrained_model=params['model_path']
        self.classifier_head=classifier_head
        self.class_size = params['num_classes']
        super(Discriminator, self).__init__()
        if self.pretrained_model.startswith("gpt2"):
            self.tokenizer = tokenizer
            self.encoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model)
            self.encoder.resize_token_embeddings(len(tokenizer))

#             # fix model padding token id
            self.encoder.config.pad_token_id = self.encoder.config.eos_token_id
            self.embed_size = self.encoder.transformer.config.hidden_size
        elif self.pretrained_model.startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.encoder = BertModel.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.config.hidden_size
        else:
            raise ValueError(
                "{} model not yet supported".format(pretrained_model)
            )
        if classifier_head:
            self.classifier_head = classifier_head
        else:
            if not self.class_size:
                raise ValueError("must specify class_size")
            self.classifier_head = ClassificationHead(
                class_size=self.class_size,
                embed_size=self.embed_size
            )
        self.cached_mode = False
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x,mask):
#         mask = mask.unsqueeze(2).repeat(
#             1, 1, self.embed_size
#         ).float().to(self.device).detach()
        mask = mask.unsqueeze(2).repeat(
            1, 1, self.embed_size
        )
        
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            #hidden, _ = self.encoder.transformer(x)
            hidden_ = self.encoder.transformer(x)
            hidden = hidden_.last_hidden_state
        else:
            # for bert
            hidden, _ = self.encoder(x)
            
        
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + self.EPSILON
        )
        return avg_hidden

    def forward(self, x, mask, labels):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x,mask)

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)
        loss=None
        if labels is not None:
            loss = F.nll_loss(probs, labels)
        
        return probs,loss

    def predict(self, input_sentence):
        input_t = self.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        mask = torch.tensor([[1]*len(input_t)], dtype=torch.long, device=self.device)
        
        if self.cached_mode:
            input_t = self.avg_representation(input_t,mask)

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob
