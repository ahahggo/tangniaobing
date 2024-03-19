import preProcessing, wordEmbedding, model
from sklearn.metrics import classification_report
from datetime import datetime
# from predict import predict


date = datetime.now().strftime("%m%d")
# from sklearn import report

embedding_dim = 300  # word2vec向量维度
batch_size = 64
num_epochs = 5
n_classes = 6
filter_sizes = [3, 6, 9]
dropout_p = 0.02
learning_rate = 0.005
num_filters = 100
# 获取分类名称
categories = []
with open('data/class.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 提取每行的字符部分，忽略数字和空格
        # 这里假设字符和数字之间至少有一个空格
        category = line.split('   ')[0]  # 根据具体的分隔符（这里假设是三个空格）分割
        categories.append(category)
print(categories)

if '__main__' == __name__:
    split_test, split_dev, split_train, train_label, dev_label = preProcessing.pre_processing()
    word2vec_model, vacab_size = wordEmbedding.embedding(split_test, split_dev, split_train, size=embedding_dim)
    train_data, dev_data, test_data = wordEmbedding.data_loader(split_test, split_dev, split_train, train_label,
                                                                dev_label,
                                                                word2vec_model, w2v_size=embedding_dim,
                                                                batch_size=batch_size)
    checkpoint_save_path = 'checkpoint/checkpoint_'+date+'.pt'
    model.train(train_data = train_data, dev_data = dev_data,
                vacab_size=vacab_size, embedding_dim=embedding_dim, n_classes=n_classes,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                dropout_p=dropout_p,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                checkpoint_save_path=checkpoint_save_path)

    # predict test data
    model_class = model.TextCNNWithAttention(vacab_size, embedding_dim,
                                       num_filters, filter_sizes,
                                       n_classes, dropout_p, hidden_size=200)

    # 对验证集进行预测，并且打印report到output文件夹下
    pred_label, pred_prob = model.predict(dev_data, model_class, checkpoint_save_path,False)
    report = classification_report(list(pred_label),dev_label,target_names=categories)
    # 保存report 到outputs文件夹下
    save_reports_path= 'outputs/dev_data_'+date+'.txt'
    with open(save_reports_path, "w") as f:
        f.write(report)
    # 对test data进行预测
    # pred_label, pred_prob = model.predict(test_data, model_class, checkpoint_save_path)
