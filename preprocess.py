import re
import random
import numpy as np

class Preprocess_tool():
    def __init__(self, pos_path, neg_path, Proportion, write_path = "../data", seed=5, sample_rate = 0.01):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.Proportion = Proportion
        self.write_path = write_path
        self.word2id = {}
        self.id_freq = {}
        self.total_words_num = 0
        self.sample_rate = sample_rate
        self.orgin_data = []
        random.seed(seed)
        np.random.seed(seed)

    def str2word_lst(self, str):
        lst = str.split()
        lst = [re.sub('[^\u4e00-\u9fa5]+', '', i) for i in lst if re.sub('[^\u4e00-\u9fa5]+', '', i) != ""]
        return lst
        
    def word_lst2id_lst(self, words, is_train = True):
        res = []
        for word in words:
            if word in self.word2id:
                res.append(self.word2id[word])
            elif is_train:
                self.id_freq[len(self.word2id)] = 0
                self.word2id[word] = len(self.word2id)
                res.append(self.word2id[word])
            
            if is_train:
                self.id_freq[self.word2id[word]] += 1
                self.total_words_num += 1
                
        return res

    def subsample(self, train_data):
        for word_tokens, _ in train_data:
            del_lst = []
            for i, word in enumerate(word_tokens):
                prob = 1 - np.sqrt(self.sample_rate / self.id_freq[word])
                # print(prob)
                sampling = np.random.sample()
                # print(sampling, prob)
                if (sampling < prob):
                    del_lst.append(i)
                
            for i in reversed(del_lst):
                del word_tokens[i]

    def preprocess(self, subsample = False):
        assert (sum(self.Proportion) == 1)
        
        data = []

        with open(self.pos_path, "r") as f1:
            pos_data = f1.readlines()

        with open(self.neg_path, "r") as f2:
            neg_data = f2.readlines()

        for i in pos_data:
            data.append([self.str2word_lst(i), 1])
            self.orgin_data.append(self.str2word_lst(i))
        print("Positive语料数:",len(pos_data))
        print("Negative语料数:",len(neg_data))
        for i in neg_data:
            data.append([self.str2word_lst(i), 0])
            self.orgin_data.append(self.str2word_lst(i))
            # print([self.str2word_lst(i), 0])

        random.shuffle(data)

        train_end, valid_start = int(len(data) * self.Proportion[0]), int(len(data) * self.Proportion[0])
        valid_end, test_start = int(len(data) * (self.Proportion[0] + self.Proportion[1])), int(len(data) * (self.Proportion[0] + self.Proportion[1]))
        
        train_data, valid_data, test_data = [], [], []
        
        
        for i in data[ : train_end]:
            train_data.append([self.word_lst2id_lst(i[0]), i[1]])

        for i in data[valid_start: valid_end]:
            valid_data.append([self.word_lst2id_lst(i[0], False), i[1]])

        for i in data[test_start: ]:
            test_data.append([self.word_lst2id_lst(i[0], False), i[1]])

        print("句子平均词数:",sum(list(self.id_freq.values())) / 1619)

        for i in self.id_freq:
            self.id_freq[i] /= self.total_words_num
            self.id_freq[i] = self.id_freq[i] ** (3./4.)

        # print(sum([len(i[0]) for i in train_data]))

        if subsample:
            self.subsample(train_data)
        print("训练集语料数:",len(train_data))
        print("验证集语料数:",len(valid_data))
        print("测试集语料数:",len(test_data))
        # print(sum([len(i[0]) for i in train_data]))
        self.write_words()

        return train_data, valid_data, test_data, self.id_freq

    def write_words(self):
        assert(self.write_path)
        with open(self.write_path + "/word.txt","w") as f:
            for i in self.word2id:
                f.write(i + "  " + str(self.word2id[i]) + "  " + str(self.id_freq[self.word2id[i]]) + '\n')

    


if __name__ == '__main__':
    pre_tool = Preprocess_tool("data/positive.txt", "data/negative.txt", [0.7, 0.1, 0.2], "./data")
    pre_tool.preprocess(subsample=True)
    pre_tool.write_words()
    pass