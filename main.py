import preProcessing
import wordEmbedding
import model

split_test, split_dev, split_train,train_label,dev_label = preProcessing.pre_processing()
word2vec_model = wordEmbedding.embedding(split_test, split_dev, split_train, size=200)
train_data,dev_data,test_data=wordEmbedding.data_loader(split_test, split_dev, split_train,train_label,dev_label, word2vec_model)
model.train(train_data)

