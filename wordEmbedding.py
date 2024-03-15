# 引入 word2vec
import numpy as np
from gensim.models import word2vec

# 引入日志配置
import logging
import pandas as pd

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data):
        self.data = data

    # 返回数据集大小
    def __len__(self):
        return self.data.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]


def embedding(test, dev, train, size=100):
    try:
        model = word2vec.Word2Vec.load('./model/word2vec' + str(size) + '.bin')
    except:
        sentences = test + dev + train
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # 构建模型
        model = word2vec.Word2Vec(sentences, min_count=1, vector_size=size)
        model.save('./model/word2vec' + str(size) + '.bin')
    return model.wv


def data_loader(test, dev, train, train_label, dev_label, model):
    padding=15
    train_data = []
    for sentence in range(len(train)):
        temp = []
        for word in train[sentence]:
            temp.append(list(model[word]))
        for i in range(padding-len(temp)):
            temp.append(list(np.zeros(200)))

        train_data.append(temp)


    dev_data = []
    for sentence in range(len(dev)):
        temp = []
        for word in dev[sentence]:
            temp.append(model[word])
        for i in range(padding-len(temp)):
            temp.append(list(np.zeros(200)))
        dev_data.append(temp)
    test_data = []
    for sentence in range(len(test)):
        temp = []
        for word in test[sentence]:
            temp.append(model[word])
        for i in range(padding-len(temp)):
            temp.append(list(np.zeros(200)))
        test_data.append(temp)


    x=torch.tensor(train_data)
    y = torch.tensor(train_label)
    dataset = TensorDataset(x,y)
    train_data = DataLoader(dataset, batch_size=64)
    print(train_data)
