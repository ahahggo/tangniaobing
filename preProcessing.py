# 文本预处理
import jieba


def read_data():
    dev = []
    test = []
    train = []
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
    return test, dev, train


def split_words(test, dev, train, stop_word):
    split_test = []
    for i in range(len(test)):
        temp = []
        for j in jieba.cut(test[i][0]):
            if j not in stop_word:
                temp.append(j)
        split_test.append(temp)
    split_dev = []
    for i in range(len(dev)):
        temp = []
        for j in jieba.cut(dev[i][0]):
            if j not in stop_word:
                temp.append(j)
        split_dev.append(temp)
    split_train = []
    for i in range(len(train)):
        temp = []
        for j in jieba.cut(train[i][0]):
            if j not in stop_word:
                temp.append(j)
        split_train.append(temp)
    return split_test, split_dev, split_train


def stop_words():
    stop_word = []
    with open("./data/stop_words.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            stop_word.append(line)

    return stop_word


def pre_processing():
    test, dev, train = read_data()  # 读取文件
    stop_word = stop_words()
    split_test, split_dev, split_train = split_words(test, dev, train, stop_word)  # 分词

    # todo:
    #     停用词
    #     数据增强
    return split_test, split_dev, split_train