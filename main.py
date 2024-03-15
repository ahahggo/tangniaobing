import preProcessing
import wordEmbedding
split_test, split_dev, split_train=preProcessing.pre_processing()
wordEmbedding.embedding(split_test, split_dev, split_train,size=100)
