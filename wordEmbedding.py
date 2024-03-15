# 引入 word2vec
from gensim.models import word2vec

# 引入日志配置
import logging


def embedding(test, dev, train, size=100):
    try:
        model = word2vec.Word2Vec.load('./model/word2vec' + str(size) + '.bin')
    except:
        sentences = test + dev + train
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # 构建模型
        model = word2vec.Word2Vec(sentences, min_count=5, vector_size=size)
        model.save('./model/word2vec' + str(size) + '.bin')

    print(model.wv['糖尿病'])

    # 进行相关性比较
    print(model.wv.similarity('糖尿病', '高血糖'))
    print(model.wv.most_similar('喝'))
