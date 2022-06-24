import os
import time
import torch
import argparse

import sys 
sys.path.append("..") 
from preprocess import Preprocess_tool
from utils import Skip_gram_dataset
from model import Skip_gram_model

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--pos_data_path', default='../data/positive.txt', type=str)
parser.add_argument('--neg_data_path', default='../data/negative.txt', type=str)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epoch_num', default=3, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--neg_num', default=10, type=int)
parser.add_argument('--W_size', default=3, type=int)
parser.add_argument('--t', default=0.001, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gpu', default="6", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == '__main__':
    pre_tool = Preprocess_tool(args.pos_data_path, args.neg_data_path, [0.7, 0.1, 0.2], sample_rate=args.t)

    train_data, valid_data, test_data, word_freq = pre_tool.preprocess(subsample=True)
    vocab_size = len(word_freq)

    train_dataset = Skip_gram_dataset(train_data, word_freq,neg_num=args.neg_num, W_size=args.W_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

    model = Skip_gram_model(vocab_size, args.embedding_dim).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    start = time.time()

    for epoch in range(args.epoch_num):
        for step, i in enumerate(train_loader):
            center_word, pos_word, neg_word = i
            center_word = center_word.to(args.device)
            pos_word = pos_word.to(args.device)
            neg_word = neg_word.to(args.device)

            optimizer.zero_grad()
            loss = model(center_word, pos_word, neg_word)
            loss.backward()
            optimizer.step()   

        print(epoch, loss.item())
        torch.save(model.state_dict(), 'save.pt')
    
    end = time.time()
    print(end - start, 's')

            
        
