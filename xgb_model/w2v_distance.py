# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import datetime
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
# from gensim.utils import tokenize
from gensim.similarities import MatrixSimilarity
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy.spatial import distance
import ipdb

import config

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def tokenize(text):
    return text.strip().split()


# glove2word2vec("vectors.txt", "glove_vectors.txt")
model = KeyedVectors.load_word2vec_format("glove_vectors.txt", binary=False)
vocab = model.vocab
train = pd.read_csv(config.path_train_cut, sep="\t", header=None, names=["id", "s1", "s2", "label"], encoding="utf-8")
tfidf_txt = train["s1"].tolist() + train["s2"].tolist()


# def get_weight(count, eps=10000, min_count=0):
#     if count < min_count:
#         return 0
#     else:
#         return 1 / (count + eps)


# texts = [tokenize(text) for text in tfidf_txt]
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
# documents = [[token for token in text if frequency[token] > 1] for text in texts]

# ipdb.set_trace()
# weights = {word: get_weight(count) for word, count in counts.items()}

# dictionary = Dictionary(documents)
# dictionary.save_as_text("./model/words.dic")
dictionary = Dictionary.load_from_text("./model/words.dic")


# class MyCorpus(object):
#     def __init__(self, documents):
#         self.documents = documents
#
#     def __iter__(self):
#         for doc in self.documents:
#             doc = tokenize(doc)
#             yield dictionary.doc2bow(doc)


# corpus = MyCorpus()
# MmCorpus.serialize("./model/corpus.mm", corpus)
# corpus = MmCorpus("./model/corpus.mm")
# tfidf = TfidfModel(corpus)
# tfidf.save("./model/tf_idf.model")
tfidf = TfidfModel.load("./model/tf_idf.model")
docs = ["花呗 分期 古天乐", "花呗 分期 查询"]

corpus = [dictionary.doc2bow(tokenize(text)) for text in docs]
print(corpus[0])
print(tfidf[corpus[0]])
ipdb.set_trace()

def tfidf_w(token):
    t2i = dictionary.token2id
    if t2i.get(token):
        res = tfidf.idfs[t2i[token]]
    else:
        res = 1.0
    return res


def eucldist_vectorized(word1, word2):
    try:
        w2v1 = model[word1]
        w2v2 = model[word2]
        sim = np.sqrt(np.sum(np.array(w2v1) - np.array(w2v2)) ** 2)
        return float(sim)
    except:
        return float(0.)


def getDiff(ws1, ws2):
    wordlist1 = ws1.split()
    wordlist2 = ws2.split()
    num = len(wordlist1) + 0.001
    sim = 0.0
    for word1 in wordlist1:
        dis = 0.0
        for word2 in wordlist2:
            if dis == 0.0:
                dis = eucldist_vectorized(word1, word2)
            else:
                dis = min(dis, eucldist_vectorized(word1, word2))
        sim += dis
    return sim / num


def getDiff_weight(ws1, ws2):
    wordlist1 = ws1.split()
    wordlist2 = ws2.split()
    tot_weights = 0.0
    for w1 in wordlist1:
        tot_weights += weights[w1]
    sim = 0.0
    for w1 in wordlist1:
        dis = 0.0
        for w2 in wordlist2:
            if dis == 0.0:
                dis = eucldist_vectorized(w1, w2)
            else:
                dis = min(dis, eucldist_vectorized(w1, w2))
        sim += weights[w1] * dis
    return sim


def getDiff_weight_tfidf(ws1, ws2):
    wordlist1 = ws1.split()
    wordlist2 = ws2.split()
    # tot_weights = 0.0
    # for w1 in wordlist1:
    #     tot_weights += tfidf_w(w1)
    sim = 0.0
    for w1 in wordlist1:
        dis = 0.0
        for w2 in wordlist2:
            if dis == 0.0:
                dis = eucldist_vectorized(w1, w2)
            else:
                dis = min(dis, eucldist_vectorized(w1, w2))
            sim += tfidf_w(w1) * dis
    return sim


def getDiff_average(ws1, ws2):
    return getDiff_weight(ws1, ws2) + getDiff_weight(ws2, ws1)


def getDiff_average_tfidf(ws1, ws2):
    return getDiff_weight_tfidf(ws1, ws2) + getDiff_weight_tfidf(ws2, ws1)


def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(list(tokenize(text)))]
    return res


def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1], num_features=len(dictionary))
    sim = index[tfidf2]
    return float(sim[0])


def get_vector(text):
    res = np.zeros([100])
    count = 0
    for word in tokenize(text):
        if word in vocab:
            res += weights[word] * model[word]
            count += weights[words]
    if count != 0:
        return res / count
    return np.zeros([100])


def get_vector_tfidf(text):
    res = np.zeros([100])
    count = 0
    for word in tokenize(text):
        if word in vocab:
            res += tfidf_w(word) * model[word]
            count += tfidf_w(word)
    if count != 0:
        return res / count





if __name__ == "__main__":
    pass












