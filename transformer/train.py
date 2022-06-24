import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import sys 
sys.path.append("..") 
from preprocess import Preprocess_tool
from utils import Dataset, evaluate
from model import transformer_model


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--pos_data_path', default='../data/positive.txt', type=str)
parser.add_argument('--neg_data_path', default='../data/negative.txt', type=str)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epoch_num', default=100, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--num_layers', default=4, type=int)
parser.add_argument('--update_emb', default=False, type=str2bool)
parser.add_argument('--n_head', default=8, type=int)
parser.add_argument('--use_pretrained', default=True, type=str2bool)
parser.add_argument('--from_gen', default=True, type=str2bool)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gpu', default="1", type=str)
parser.add_argument('--log_name', default="", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == '__main__':
    if args.log_name == "":
        if not args.use_pretrained:
            args.log_name += "not_"
        args.log_name+="use_pretrained, "

    f = open(args.log_name,"w")
    pre_tool = Preprocess_tool(args.pos_data_path, args.neg_data_path, [0.7, 0.1, 0.2])

    train_data, valid_data, test_data, word_freq = pre_tool.preprocess(subsample=False)
    # print(len(train_data), len(valid_data), len(test_data))
    # print("update_emb:",args.update_emb)

    pad_word = len(word_freq)
    vocab_size = len(word_freq)
    train_dataset = Dataset(train_data, pad_word) 
    test_dataset = Dataset(test_data, pad_word) 
    valid_dataset = Dataset(valid_data, pad_word) 

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)

    if args.use_pretrained:
        if args.from_gen:
            print('from gen:')
            pretrained_weight = torch.load("../word_emb/gen.pt").to(args.device)
            pad_emb = torch.zeros([1,128]).to(args.device)
            pretrained_weight = torch.cat([pretrained_weight, pad_emb], dim = 0).to(args.device)
        else:
            print('from train:')
            pretrained_weight = torch.load("../word_emb/save.pt")['in_embedding.weight'].to(args.device)
            pad_emb = torch.zeros([1,128]).to(args.device)
            pretrained_weight = torch.cat([pretrained_weight, pad_emb], dim = 0).to(args.device)
    else:
        print("no_pretrain:")
        pretrained_weight = None

    model = transformer_model(vocab_size, args.embedding_dim, pretrained_weight, update_emb = args.update_emb, 
                 num_layers = args.num_layers, n_head = args.n_head).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    start = time.time()

    criterion = nn.CrossEntropyLoss()

    best_valid = (0,0,0,0)
    best_test = (0,0,0,0)

    for epoch in range(args.epoch_num):
        all_loss = 0
        for step, i in enumerate(train_loader):
            x, y = i
            y = y.reshape(-1)
            out = model(x.to(args.device))
            loss = criterion(out, y.to(args.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()

        model.eval()
        valid_eval = evaluate(valid_dataset, model, args)
        test_eval = evaluate(test_dataset, model, args)
        if valid_eval[0] > best_valid[0]:
            best_test = test_eval
        f.write(str(valid_eval[0]) + " " + str(test_eval[0]) + "\n")
        # print(epoch, evaluate(valid_dataset, model, args), evaluate(test_dataset, model, args))
        model.train()
        
    print("Accuracy:", best_test[0])
    print("Precision:", best_test[1])
    print("Recall:", best_test[2])
    print("F1:", best_test[3])
    f.close()
            