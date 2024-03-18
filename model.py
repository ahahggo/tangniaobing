import torch
import torch.nn as nn

import torch.nn.functional as F
# 假设您已经有了预处理过的数据集，并且已经通过Word2Vec生成了词向量
# 词向量维度假设为EMBEDDING_DIM
EMBEDDING_DIM = 200  # 根据实际情况替换为实际的词向量维度大小
VOCAB_SIZE = 5569  # word2vec词汇表大小
N_CLASSES = 6  # 类别数量，根据您的多分类任务定义


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout_p):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 如果有预训练好的词向量，则可以加载到embedding层

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text):
        embedded = text

        conved = [F.relu(conv(embedded.permute(0, 2, 1))) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = torch.cat(pooled, dim=1)
        dropout = self.dropout(cat)

        return self.fc(dropout)


def train(train_data):
    # 参数设定
    num_filters = 100  # 卷积核数量
    filter_sizes = [3, 4, 5]  # 不同长度的卷积窗口大小
    dropout_p = 0.5  # Dropout比例

    model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, num_filters, filter_sizes, N_CLASSES, dropout_p)

    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 使用DataLoader进行训练和验证
    # ...（此处省略DataLoader的实例化与训练循环代码）

    # 训练过程中的一个典型迭代步骤：
    for batch in train_data:
        inputs, labels = batch
        if not inputs.dtype == torch.long:
            inputs = inputs.to(torch.long)
        inputs = inputs.to(torch.float)

        optimizer.zero_grad()

        outputs = model(inputs)  # 直接使用原始的词向量序列作为输入
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct = (outputs.argmax(dim=1) == labels)
        accuracy = correct.sum() / len(labels)
        print('acc: ',accuracy.item(), 'loss: ',loss.item())
