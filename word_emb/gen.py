import os
import time
from gensim.models import Word2Vec
import argparse
import torch

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
    train_data, valid_data, test_data, word_freq =pre_tool.preprocess()
    # data = pre_tool.orgin_data

    data = [i[0] for i in train_data]
    
    print(data[:5])

    model = Word2Vec(sentences=data, vector_size=128, window=3, min_count=1, workers=4)

    model.train(data, total_examples=model.corpus_count, epochs=30)

    emb = []

    for i in word_freq:
        emb.append(torch.tensor([model.wv[i]]))

    weight = torch.cat(emb, dim=0)

    print(weight.shape)
    torch.save(weight, 'gen.pt')




