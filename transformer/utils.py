import torch

class Dataset():
    def __init__(self, data, padword, maxlen = 20):
        self.orgin_data = data
        self.data = data  
        for idx, [sentence, _] in enumerate(data):
            sentence = sentence[:maxlen]
            sentence = [padword for i in range(maxlen - len(sentence))] + sentence
            data[idx][0] = sentence

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx][0])
        return torch.LongTensor(self.data[idx][0]), torch.LongTensor([self.data[idx][1]])

def evaluate(dataset, model, args):
    x = [i[0] for i in dataset.data]
    y = [i[1] for i in dataset.data]
    y = torch.LongTensor(y).to(args.device)
    x = torch.LongTensor(x).to(args.device)
    predict = model(x).argmax(dim = 1)
    # print(predict.shape, y.shape)
    
    TP = ((predict == y) * y).sum().item()
    TN = ((predict == y) * (y ^ 1)).sum().item()
    FP = ((predict != y) * (y ^ 1)).sum().item()
    FN = ((predict != y) * y).sum().item()

    # print(TP, TN, FP,FN)
    

    Acc = (TP + TN) / (TP + TN + FP + FN)
    if (TP + FP == 0):
        Pre = 0
    else:
        Pre = TP / (TP + FP)
    Rec = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)
    

    return Acc, Pre, Rec, F1