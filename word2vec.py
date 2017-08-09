# coding: utf-8

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.ldamodel import LdaModel
from sklearn.cluster import KMeans
from janome.tokenizer import Tokenizer


import math
import numpy
import logging
from copy import copy

SAMPLE_WORDS = ["おいしい", "美味しい", "苦い", "辛い","赤い", "青い", "黄色い", "茶色い", "寂しい", "嬉しい", "悔しい", "恋しい", "寒い", "暑い"]
SAMPLE_WORD_GROUPS = {"味": ["おいしい", "美味しい", "苦い", "辛い"],
                      "色": ["赤い", "青い", "黄色い", "茶色い"],
                      "感情": ["寂しい", "嬉しい", "悔しい", "恋しい"],
                      "温度": ["寒い", "暑い", ]}

class WordModel(object):
    model_path = ''

    def __init__(self):
        self.model = Word2Vec.load(self.model_path)


class EnglishModel(WordModel):
    model_path = 'model/english_w2v.model'

    def __init__(self):
        super(EnglishModel, self).__init__()


class JapaneseModel(WordModel):
    model_path = 'model/japanese_w2v.model'

    def __init__(self):
        super(JapaneseModel, self).__init__()


def categorize(target_word, groups=SAMPLE_WORD_GROUPS):

    average_similarities = []
    jm = JapaneseModel()
    for name, words in groups.items():
        similarities = [jm.model.similarity(word, target_word) for word in words]
        average_similarity = sum(similarities) / float(len(similarities))
        average_similarities.append((name, average_similarity))

    return sorted(average_similarities, key=lambda x:x[1], reverse=True)


def clustering(n_clusters, target_words=SAMPLE_WORDS):
    jm = JapaneseModel()

    word_vectors = numpy.array([jm.model.wv[word] for word in target_words])
    k_means = KMeans(n_clusters=n_clusters, random_state=150)

    result = list(k_means.fit_predict(word_vectors))

    cluster_dict = {i: [] for i in range(n_clusters)}
    for cluster, word in zip(result, target_words):
        cluster_dict[cluster].append(word)

    return cluster_dict

def extract_noun(sentence):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(sentence)

    for token in tokens:
        lst = categorize(token.surface)
        if lst[0][1] > 0.6:
            category_string = lst[0][0]
        else:
            category_string = "N/A"

        print('{0} : {1}'.format(token.surface, category_string))


def load_csv(filename="data/stream.csv"):
    f = open(filename, 'r')
    for line in f:
        data = line.split(',')
        print(data[1])
