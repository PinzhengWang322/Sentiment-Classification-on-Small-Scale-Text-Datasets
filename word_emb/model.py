import torch
import torch.nn as nn
import torch.nn.functional as F

class Skip_gram_model(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(Skip_gram_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize both embedding tables with uniform distribution
        self.in_embedding.weight.data.uniform_(-1,1)
        self.out_embedding.weight.data.uniform_(-1,1)

    def forward(self, center_word, pos_word, neg_word):
        
        batch_size = center_word.shape[0]
        center_emb = self.in_embedding(center_word)
        pos_emb = self.out_embedding(pos_word).permute(0,2,1)
        neg_emb = self.out_embedding(neg_word).permute(0,2,1)

        pos_loss = torch.bmm(center_emb, pos_emb)
        neg_loss = - torch.bmm(center_emb, neg_emb)

        pos_loss = F.logsigmoid(pos_loss).sum(1)
        neg_loss = F.logsigmoid(neg_loss).sum(1)

        total_loss = - (pos_loss + neg_loss).mean()

        return total_loss



