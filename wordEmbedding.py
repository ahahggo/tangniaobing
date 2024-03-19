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
    return model.wv,len(model.wv.index_to_key)


def data_loader(test, dev, train, train_label, dev_label, model, padding=15,w2v_size=100, batch_size=64):
    def pad_and_transform(data, padding_size):
        temp = []
        for sentence in data:
            padded_sentence = [list(model[word]) for word in sentence]
            while len(padded_sentence) < padding_size:
                padded_sentence.append(list(np.zeros(w2v_size)))
            temp.append(np.array(padded_sentence))
        # 返回二维numpy数组，而不是包含多个一维数组的列表
        return np.stack(temp)

    # 一次处理训练、验证和测试数据
    train_data = pad_and_transform(train, padding)
    dev_data = pad_and_transform(dev, padding)
    test_data = pad_and_transform(test, padding)

    # 将numpy数组转换为torch张量
    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_label)
    train_dataset = TensorDataset(x_train, y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)

    x_dev = torch.tensor(dev_data, dtype=torch.float32)
    y_dev = torch.tensor(dev_label)
    dev_dataset = TensorDataset(x_dev, y_dev)
    dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size)

    x_test = torch.tensor(test_data, dtype=torch.float32)
    # test_dataset = TensorDataset(x_test)
    test_data_loader = DataLoader(x_test, batch_size=batch_size)

    return train_data_loader, dev_data_loader, test_data_loader
