import preProcessing
import wordEmbedding
import model
# from sklearn import report

word2vec_size = 300  # word2vec向量维度
batch_size = 64
num_epochs = 50
n_classes = 6

if '__main__' == __name__:
    split_test, split_dev, split_train, train_label, dev_label = preProcessing.pre_processing()
    word2vec_model, vocab_num = wordEmbedding.embedding(split_test, split_dev, split_train, size=word2vec_size)
    train_data, dev_data, test_data = wordEmbedding.data_loader(split_test, split_dev, split_train, train_label,
                                                                dev_label,
                                                                word2vec_model, w2v_size=word2vec_size,
                                                                batch_size=batch_size)
    model.train(train_data, dev_data, embedding_dim=word2vec_size, vacab_size=vocab_num, n_classes=n_classes,
                num_epochs=num_epochs)
