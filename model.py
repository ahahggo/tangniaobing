import torch
import torch.nn as nn

import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy stopped improving.
                            Default: 3
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = float('-inf')
        self.delta = delta

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation accuracy increases."""
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')  # Save the model
        self.val_acc_max = val_acc

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

def evaluate(model, data_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():  # No gradients needed
        for inputs, labels in data_loader:
            inputs = inputs.to(torch.float)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted_labels = torch.max(outputs, dim=1)
            correct = (predicted_labels == labels).float()
            total_accuracy += correct.sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples

    return avg_accuracy, avg_loss

def train(train_data, dev_data, embedding_dim,  # 根据实际情况替换为实际的词向量维度大小
          vacab_size,  # word2vec词汇表大小
          n_classes,  # 类别数量，根据您的多分类任务定义
          num_epochs=10
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
    # 初始化earlystopping
    early_stopping = EarlyStopping(patience=10, delta = 0.001, verbose=False)

    for epoch in range(num_epochs):
        for inputs, labels in train_data:
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
        dev_accuracy, dev_loss = evaluate(model, dev_data, criterion)
        print(f'Epoch [{epoch + 1}/{num_epochs}], , Training Loss: {avg_loss:.4f}, Dev Acc: {dev_accuracy * 100:.2f}%')
        # 调用early_stopping对象进行检查，注意这里传递的是dev_accuracy

        early_stopping(dev_accuracy, model)

        # 检查是否需要早停
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # 重置累计损失、准确率和样本数以准备下一个epoch
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
