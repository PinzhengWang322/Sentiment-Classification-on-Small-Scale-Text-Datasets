import numpy as np
import torch
import random

import sys 
sys.path.append("..") 
from preprocess import Preprocess_tool

class Skip_gram_dataset():
    def __init__(self, data, word_freq, neg_num = 10, W_size = 3, batch_size = 64, seed = 5):
        self.orgin_data = data
        self.neg_num = neg_num
        self.W_size = W_size
        self.word_freq = torch.tensor(list(word_freq.values()))
        self.batch_size = 64
        self.data = []   
        torch.manual_seed(seed)

        self.load_data()

    def load_data(self):
        for word_tokens, _ in self.orgin_data:
            for idx, center_word in enumerate(word_tokens):
                for pos_word in word_tokens[max(0, idx - self.W_size) : idx + self.W_size + 1]:
                    if (pos_word != center_word): 
                        self.data.append([center_word, pos_word])   
                    
                    # neg_words = torch.multinomial(self.word_freqs, 1, True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        neg_words = torch.multinomial(self.word_freq, self.neg_num, True)

        while self.data[idx][0] in neg_words or self.data[idx][1] in neg_words:
            neg_words = torch.multinomial(self.word_freq, self.neg_num, True)

        return torch.LongTensor([self.data[idx][0]]), torch.LongTensor([self.data[idx][1]]), neg_words

    

if __name__ == '__main__':
    pre_tool = Preprocess_tool("../data/positive.txt", "../data/negative.txt", [0.7, 0.1, 0.2], "./data")
    
    train_data, valid_data, test_data, word_freq = pre_tool.preprocess(subsample=True)

    dataset = Skip_gram_dataset(train_data, word_freq)

    print(train_data[0])
    print(dataset.data[:5])
    print(dataset[0])
    print(dataset[0])
    print(dataset[1])


    