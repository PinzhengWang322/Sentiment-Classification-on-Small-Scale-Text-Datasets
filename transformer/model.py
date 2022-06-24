import torch
import torch.nn as nn
import torch.nn.functional as F
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class transformer_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight, update_emb, 
                 n_head, num_layers):
        super(transformer_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        if pretrained_weight != None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        else:
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx = vocab_size,)

        self.embedding.weight.requires_grad = update_emb

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head)

        self.norm = LayerNorm(embedding_dim)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm = self.norm)

        self.decoder = nn.Linear(embedding_dim, 2)
        
    def forward(self, inputs):
       
        embeddings = self.embedding(inputs)
        
        encoding = self.encoder(embeddings.permute([1, 0, 2])).permute([1, 0, 2])

        encoding = encoding.sum(dim = 1)

        outputs = self.decoder(encoding)
        

        return outputs

        
