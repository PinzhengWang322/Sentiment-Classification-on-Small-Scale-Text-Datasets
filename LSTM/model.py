import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight, update_emb, hidden_dim,
                 num_layers, dropout, bidirectional,):
        super(LSTM_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        if pretrained_weight != None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        else:
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx = vocab_size,)

        self.embedding.weight.requires_grad = update_emb
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=dropout)

        if self.bidirectional:
            self.linear1 = nn.Linear(4 * hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 2)
        else:
            self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, 2)


        
    def forward(self, inputs):
       
        embeddings = self.embedding(inputs)
        
        hn, cn = self.encoder(embeddings.permute([1, 0, 2]))
       
        encoding = torch.cat([hn[0], hn[-1]], dim=1)
        
        outputs = self.linear1(encoding)
        
        outputs = self.linear2(outputs)

        return outputs

        
