import os.path
import sys
import openpyxl
import pandas as pd
import torch,time
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
date = datetime.now().strftime("%m%d")

import os.path
import sys
import openpyxl
import pandas as pd
import random
import torch, time
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

date = datetime.now().strftime("%m%d")


# 定义自己的Dataset
class ClassDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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


def pred_func(data_loader, model_path, num_labels, checkpoint_save_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    # 加载训练好的权重
    # 移动模型到合适的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_save_path, map_location=device))
    # model.load_state_dict(torch.load(checkpoint_save_path))
    # 确保模型在评估模式
    model.eval()
    predictions, probabilities = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            # 获取每个样本的最大概率值及其索引
            max_probs, preds = torch.max(probs, dim=1)
            #             predictions = torch.argmax(logits,dim=1).cpu().numpy()
            predictions.extend(preds.tolist())
            probabilities.extend(max_probs.tolist())
    return predictions, probabilities


def train_model(train_loader, dev_loader,
                model_path, num_labels,
                checkpoint_save_path='checkpoint/bert_checkpoint.pth',
                num_epochs=50,
                learning_rate=1e-5,
                patience=5,
                delta=0.001):
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.train()  # 设置模型维训练模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    epoch_loss = []
    best_epoch_index = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_total = 0.0
        train_correct = 0.0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            train_total += labels.size(0)
            train_correct += (predictions == labels).sum().item()

        average_train_loss = total_loss / len(train_loader)
        epoch_loss.append(average_train_loss)
        train_accuracy = train_correct / train_total

        # 验证
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_predictions)

        # 检查是否要提前终止训练
        early_stopping(accuracy)
        if early_stopping.early_stop:
            print('Early Stopping')
            break

        # 保存最佳模型
        if epoch == 0 or accuracy > early_stopping.best_score:
            early_stopping.best_score = accuracy
            best_epoch_index = epoch + 1
            torch.save(model.state_dict(), checkpoint_save_path)
        # 打印当前epoch的信息
        print(
            f'Epoch {epoch + 1}: Training loss = {epoch_loss[epoch]:.4f},Training accuracy = {train_accuracy:.4f}, Validation accuracy = {accuracy:.4f}, Best Accuracy = {early_stopping.best_score:.4f}')

    # 打印模型最佳的准确度
    print(f'Epoch {best_epoch_index} has the best validation accuracy:{early_stopping.best_score:.4f}')
    # 打印loss 下降趋势图
    plt.plot()
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss)
    # 在最优accuracy处添加红点
    plt.scatter(best_epoch_index, epoch_loss[best_epoch_index - 1], color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epoch')
    plt.savefig('pics/loss_plot_bert_' + date + '.png')


def set_seed(seed_value=42):
    """设置所有随机种子以确保实验的可重复性。"""
    random.seed(seed_value)  # Python的随机库
    np.random.seed(seed_value)  # NumPy库
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # CUDA的随机种子
        torch.cuda.manual_seed_all(seed_value)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rm_characters(original_string, characters_to_remove):
    modified_string = original_string
    for character in characters_to_remove:
        modified_string = modified_string.replace(character, "")
    return modified_string


rm_words = ['糖尿病', '糖尿病人', '糖尿']

if __name__=="__main__":
    # ===train model + dev data prediction=======
    model_path = '/Users/yujinge/Documents/model/chinese-bert-wwm-ext'
    checkpoint_save_path = os.path.join('checkpoint', 'bert_' + date + '.pth')
    dev_save_path = 'outputs/dev_data_bert_pred_' + date + '.csv'
    save_report_path = 'outputs/dev_data_bert_pred_' + date + '.txt'
    data_path = 'data/'
    train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    dev_data = pd.read_csv(os.path.join(data_path, 'dev.csv'))
    # torch.cuda.empty_cache() # 清理缓存，释放GPU内存？
    # 获取分类名称
    label_dict = {}
    with open('data/class.txt', 'r', encoding='utf-8') as file:
        for line in file:
            tmp = line.split('   ')
            label_dict[tmp[1].replace('\n', '').strip(' ')] = tmp[0]
    print(label_dict)

    # 去除content中的糖尿病等词汇，降低相似度（结果差距不大）
    train_data_con, dev_data_con = list(train_data['content']), list(dev_data['content'])
    train_data_rm, dev_data_rm = [], []
    for items in train_data_con:
        items = rm_characters(items, rm_words)
        train_data_rm.append(items)
    for items in dev_data_con:
        items = rm_characters(items, rm_words)
        dev_data_rm.append(items)

    max_len = 32  # 最大的token长度为38
    num_epochs = 50
    batch_size = 512
    learning_rate = 1e-5
    patience = 5
    delta = 0.001
    num_labels = len(Counter(train_data['label']))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = ClassDataset(train_data_rm, train_data['label'], tokenizer, max_len)
    dev_dataset = ClassDataset(dev_data_rm, dev_data['label'], tokenizer, max_len)
    # train_dataset = ClassDataset(train_data['content'],train_data['label'], tokenizer, max_len)
    # dev_dataset = ClassDataset(dev_data['content'], dev_data['label'], tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # 对模型进行训练
    print("begin training model!")
    start_time = time.time()
    train_model(train_loader,dev_loader,model_path,num_labels,checkpoint_save_path,num_epochs,
                learning_rate, patience, delta)
    print('训练集数据量：{}，训练耗时：{}'.format(train_data.shape[0], round(time.time() - start_time, 3)))

    # 对验证集进行预测，并保留预测结果
    checkpoint_save_path = os.path.join('checkpoint', 'bert_' + date + '.pth')
    prediction, probability = pred_func(dev_loader, model_path, num_labels, checkpoint_save_path)
    dev_data['con_clean'] = dev_data_rm
    dev_data['pred_label'] = prediction
    dev_data['pred_catagory'] = dev_data['pred_label'].astype(str)
    dev_data['pred_catagory'] = dev_data['pred_catagory'].map(label_dict)
    dev_data['pred_prob'] = probability
    # 保存report 到outputs文件夹下
    report = classification_report(prediction, list(dev_data['label']))
    with open(save_report_path, "w") as f:
        f.write(report)
    res = []
    for pred, real in zip(prediction, dev_data['label']):
        if str(pred) != str(real):
            res.append(1)
        else:
            res.append(0)
    dev_data['wrong'] = res
    dev_data.to_csv(dev_save_path, index=False)

    # model_path="/Users/yujinge/Documents/model/chinese-bert-wwm-ext"
    # # 输出所有数据的tokenizer的长度
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # # tokens = tokenizer.tokenize(text)
    # token_len = []
    # for items in test['content']:
    #     tokens = tokenizer.tokenize(items)
    #     token_len.append(len(tokens))
    # print(max(token_len))









