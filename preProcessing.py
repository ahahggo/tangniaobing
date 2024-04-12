# 文本预处理
import jieba
import numpy as np
import pandas as pd


def to_one_hot(labels, num_classes):
    """
    将整数标签列表转换为one-hot编码形式。

    参数：
    labels (list or numpy.ndarray): 整数标签列表。
    num_classes (int): 类别的总数。

    返回：
    numpy.ndarray: one-hot编码后的数组，形状为 `(len(labels), num_classes)`。
    """

    # 将列表转化为numpy数组
    labels = np.array(labels)

    # 创建一个对角线为类别个数的矩阵，其中对角线元素为1，其余位置为0
    one_hot_labels = np.eye(num_classes)[labels]

    return one_hot_labels


def read_data():
    dev = []
    test = []
    train = []
    class_dict = {}
    with open("./data/test.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            test.append([line])
    with open("./data/dev.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            line = line.split('\t')
            dev.append(line)
    with open("./data/train.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            line = line.split('\t')
            train.append(line)
    with open("./data/class.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            line = line.split('   ')
            class_dict[line[1]] = line[0]
    return test, dev, train, class_dict


def split_words(test, dev, train, stop_word):
    split_test = []
    for i in range(len(test)):
        temp = []
        for j in jieba.cut(test[i][0]):
            if j not in stop_word:
                temp.append(j)
        split_test.append(temp)
    split_dev = []
    dev_label = []
    for i in range(len(dev)):
        temp = []

        dev_label.append(int(dev[i][1]))
        for j in jieba.cut(dev[i][0]):

            if j not in stop_word:
                temp.append(j)
        split_dev.append(temp)
    split_train = []
    train_label = []
    for i in range(len(train)):
        temp = []
        train_label.append(int(train[i][1]))
        for j in jieba.cut(train[i][0]):

            if j not in stop_word:
                temp.append(j)
        split_train.append(temp)
    return split_test, split_dev, split_train, train_label, dev_label


def stop_words():
    stop_word = []
    with open("./data/stop_words.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            stop_word.append(line)

    return stop_word


def pre_processing():
    test, dev, train, class_dict = read_data()  # 读取文件
    test = pd.DataFrame(test,columns=['content'])
    dev = pd.DataFrame(dev,columns=['content','label'])
    train = pd.DataFrame(train,columns=['content','label'])
    train['class'] = train['label'].map(class_dict)
    test.to_csv('data/test.csv', index=False)
    train.to_csv('data/train.csv',index=False)
    dev.to_csv('data/dev.csv',index=False)

    stop_word = stop_words()
    return split_words(test, dev, train, stop_word)  # 分词
