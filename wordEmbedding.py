# 引入 word2vec
import numpy as np
from gensim.models import word2vec

# 引入日志配置
import logging
import pandas as pd

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader


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
    padding = 15
    train_data = []
    for sentence in range(len(train)):
        temp = []
        for word in train[sentence]:
            temp.append(list(model[word]))
        for i in range(padding - len(temp)):
            temp.append(list(np.zeros(200)))

        train_data.append(temp)

    dev_data = []
    for sentence in range(len(dev)):
        temp = []
        for word in dev[sentence]:
            temp.append(model[word])
        for i in range(padding - len(temp)):
            temp.append(list(np.zeros(200)))
        dev_data.append(temp)
    test_data = []
    for sentence in range(len(test)):
        temp = []
        for word in test[sentence]:
            temp.append(model[word])
        for i in range(padding - len(temp)):
            temp.append(list(np.zeros(200)))
        test_data.append(temp)

    x = torch.tensor(train_data)
    y = torch.tensor(train_label)
    dataset = TensorDataset(x, y)
    train_data = DataLoader(dataset, batch_size=64)

    x = torch.tensor(dev_data)
    y = torch.tensor(dev_label)
    dataset = TensorDataset(x, y)
    dev_data = DataLoader(dataset, batch_size=64)

    x = torch.tensor(test_data)
    dataset = TensorDataset(x)
    test_data = DataLoader(dataset, batch_size=64)
    return train_data,dev_data,test_data
