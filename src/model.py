import torch
from torch import nn
import os



class CharModel(nn.Module):
    # A Character predicting model
    def __init__(self,
                 input_size,
                 embedding_dim,
                 hidden_size,
                 max_norm,
                 num_layers,
                 dense_size,
                 dropout = 0,
                 device = "cpu", 
                 *args, **kargs) -> None:
        super().__init__()
        self.device = device
        lstm_dropout = 0 if num_layers == 1 else dropout
        self.embeding = nn.Embedding(
            input_size, embedding_dim=embedding_dim, padding_idx=input_size-1, norm_type=2, max_norm=max_norm,
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=lstm_dropout
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, dense_size)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dense_size, input_size)
        self.to(device)
        
        
    def forward(self, x, h=None, c=None):
        # Parameters:
        #   x: input
        #   h, c: hidden state of LSTM
        x = self.embeding(x)
        if h is not None and c is not None:
            _, (h,c) = self.lstm(x, (h,c))
        else:
            _, (h,c) = self.lstm(x)
        h_mean = h.mean(dim=0)
        x = self.linear_1(h_mean)
        logits = self.linear_2(x)
        return logits, h, c
    
    def predict(self, x, h = None, c = None):
        return self(x, h, c)[0]
    
    def save(self, model_dir: str):
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        torch.save(self.state_dict(), model_dir)
    
    def load_data(self, model_dir: str):
        self.load_state_dict(torch.load(model_dir, map_location=self.device, weights_only=False))
