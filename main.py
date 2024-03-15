import preProcessing
import wordEmbedding

split_test, split_dev, split_train,train_label,dev_label = preProcessing.pre_processing()
model = wordEmbedding.embedding(split_test, split_dev, split_train, size=200)
wordEmbedding.data_loader(split_test, split_dev, split_train,train_label,dev_label, model)
