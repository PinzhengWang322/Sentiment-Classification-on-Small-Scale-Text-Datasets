import os
import time
import torch
import argparse
import torch.nn.functional as F
import numpy as np
import scipy.spatial as T

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
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch_num', default=20, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--neg_num', default=0.001, type=float)
parser.add_argument('--t', default=0.001, type=float)
parser.add_argument('--gpu', default="1", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



if __name__ == '__main__':
    pre_tool = Preprocess_tool(args.pos_data_path, args.neg_data_path, [0.7, 0.1, 0.2], sample_rate=args.t)

    train_data, valid_data, test_data, word_freq = pre_tool.preprocess(subsample=True)

    word2id = pre_tool.word2id
    id2word = list(word2id)

    vocab_size = len(word_freq)
    
    model = Skip_gram_model(vocab_size, args.embedding_dim)
    model.load_state_dict(torch.load("save.pt"))



    def find_nearest(word):
        index = word2id[word]
        embedding = model.in_embedding.weight.detach().numpy()
        word_emb = embedding[index]
        cos_dis = np.array([T.distance.cosine(e, word_emb) for e in embedding])
        print(cos_dis.shape)
        return [id2word[i] for i in cos_dis.argsort()[:10]]

    
    print(find_nearest("高兴"))
    print(find_nearest("深圳"))
