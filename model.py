import torch
import torch.nn as nn

import torch.nn.functional as F


class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy stopped improving.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation accuracy improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = float('-inf')
        self.delta = delta

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0  # reset counter if validation accuracy has improved


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (seq_len, batch, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=0)
        # context_vector shape after sum: (batch, hidden_size)
        context_vector = torch.sum(attention_weights * lstm_output, dim=0)
        return context_vector, attention_weights


class TextCNNWithAttention(nn.Module):
    def __init__(self, vacab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout_p, hidden_size):
        super(TextCNNWithAttention, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters + hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, text):
        embedded = text
        lstm_output, (hidden, cell) = self.lstm(embedded.permute(1, 0, 2))
        attention_output, _ = self.attention(lstm_output)
        conved = [F.relu(conv(embedded.permute(0, 2, 1))) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled + [attention_output], dim=1)
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

import torch

def predict(data_loader, model, checkpoint_save_path, if_test = True):
    # 加载模型
    model.load_state_dict(torch.load(checkpoint_save_path))
    model.eval()

    # 存储预测结果
    predictions = []
    probabilities = []
    if if_test:
        with torch.no_grad():  # 不更新权重
            for inputs in data_loader:
                inputs = inputs.to(torch.float)
                outputs = model(inputs)
                prob, predicted_labels = torch.max(outputs, dim=1)
                predictions.extend(predicted_labels)
                probabilities.extend(prob.squeeze().tolist())  # 压缩并转为列表形式
    else:
        with torch.no_grad():  # 不更新权重
            for inputs, labels in data_loader:
                inputs = inputs.to(torch.float)
                outputs = model(inputs)
                prob, predicted_labels = torch.max(outputs, dim=1)
                predictions.extend(predicted_labels)
                probabilities.extend(prob.squeeze().tolist())  # 压缩并转为列表形式

    return predictions, probabilities

def train(train_data, dev_data,
          embedding_dim,  # 根据实际情况替换为实际的词向量维度大小
          vacab_size,  # word2vec词汇表大小
          n_classes,  # 类别数量，根据您的多分类任务定义
          num_epochs=10,
          num_filters=100,
          filter_sizes=[3, 6, 9],
          dropout_p=0.02,
          learning_rate=0.005,
          checkpoint_save_path = 'checkpoint/checkpoint.pt' #断点储存路径
          ):
    # 参数设定

    model = TextCNNWithAttention(vacab_size, embedding_dim, num_filters, filter_sizes, n_classes, dropout_p,hidden_size=200)

    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    # 初始化earlystopping
    early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=False)

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
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Acc: {avg_accuracy * 100:.2f}% , Training Loss: {avg_loss:.4f}, Dev Acc: {dev_accuracy * 100:.2f}%')

        # Early stopping logic and save best model
        early_stopping(dev_accuracy)
        if early_stopping.best_score == dev_accuracy:
            # Save the model only when validation accuracy improves
            torch.save(model.state_dict(), checkpoint_save_path)
            # print(f"Validation accuracy improved, saving model to {checkpoint_save_path}")
        elif early_stopping.early_stop:
            print("Early stopping")
            break  # Exit from the training loop
            
        # 重置累计损失、准确率和样本数以准备下一个epoch
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
