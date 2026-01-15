import torch
import torch.nn as nn
from transformers import DistilBertModel

class BERTWithMetadata(nn.Module):
    def __init__(self, metadata_dim, dropout=0.3):
        super(BERTWithMetadata, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, metadata):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0]  # [CLS] token
        
         # Debug
        if torch.isnan(cls_token).any():
            print("NaN in cls_token")
        
        metadata_out = self.metadata_net(metadata)
        
        # Debug
        if torch.isnan(metadata_out).any():
            print("NaN in metadata_out")
            
            
        combined = torch.cat((cls_token, metadata_out), dim=1)
        
        # Debug
        if torch.isnan(combined).any():
            print("NaN in combined")
            
        output = self.classifier(combined)
        
        # Debug
        if torch.isnan(output).any():
            print("NaN in output")
            
        return output
