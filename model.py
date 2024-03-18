import torch
import torch.nn as nn

import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vacab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout_p):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vacab_size, embedding_dim)
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


def train(train_data, embedding_dim,  # 根据实际情况替换为实际的词向量维度大小
          vacab_size,  # word2vec词汇表大小
          n_classes,  # 类别数量，根据您的多分类任务定义
          num_epochs=1
          ):
    # 参数设定
    num_filters = 100
    filter_sizes = [4, 5, 6]
    dropout_p = 0.1

    model = TextCNN(vacab_size, embedding_dim, num_filters, filter_sizes, n_classes, dropout_p)

    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    for epoch in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_data):
            if not inputs.dtype == torch.long:
                inputs = inputs.to(torch.long)
            inputs = inputs.to(torch.float)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 计算并累加损失与正确预测数量
            total_loss += loss.item() * inputs.size(0)
            _, predicted_labels = torch.max(outputs, dim=1)
            correct = (predicted_labels == labels).float()
            total_accuracy += correct.sum().item()
            total_samples += inputs.size(0)

        # 每个epoch结束后计算并打印平均损失和准确率
        avg_loss = total_loss / len(train_data.dataset)
        avg_accuracy = total_accuracy / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Acc: {avg_accuracy * 100:.2f}%, Loss: {avg_loss:.4f}')

        # 重置累计损失、准确率和样本数以准备下一个epoch
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
