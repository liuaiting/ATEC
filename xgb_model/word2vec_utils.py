# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary, MmCorpus
# from gensim.utils import tokenize
from gensim.similarities import MatrixSimilarity
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy import spatial

import config

import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def tokenize(sentence):
    return sentence.strip().split()


# glove2word2vec("vectors.txt", "glove_vectors.txt")
glove_vectors_file = "xgb_model/glove_vectors.txt"
word2vec_vectors_file = "xgb_model/word2vec_vectors.txt"
word2vec_vectors = KeyedVectors.load_word2vec_format(word2vec_vectors_file, binary=False)
glove_vectors = KeyedVectors.load_word2vec_format(glove_vectors_file, binary=False)
word2vec_vocab = word2vec_vectors.vocab
glove_vocab = glove_vectors.vocab
dictionary = Dictionary.load_from_text("xgb_model/model/words.dic")
tfidf = TfidfModel.load("xgb_model/model/tf_idf.model")


def train_tfidf(inpath=config.path_train_cut):
    train_df = pd.read_csv(inpath, sep="\t", header=None, names=["id", "s1", "s2", "label"], encoding="utf-8")
    tfidf_txt = train_df["s1"].tolist() + train_df["s2"].tolist()
    texts = [tokenize(text) for text in tfidf_txt]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    documents = [[token for token in text if frequency[token] > 1] for text in texts]

    dictionary = Dictionary(documents)
    dictionary.save_as_text("./model/words.dic")
    # dictionary = Dictionary.load_from_text("./model/words.dic")

    class MyCorpus(object):
        def __iter__(self):
            for doc in documents:
                yield dictionary.doc2bow(doc)

    corpus = MyCorpus()
    MmCorpus.serialize("./model/corpus.mm", corpus)
    # corpus = MmCorpus("./model/corpus.mm")
    tfidf = TfidfModel(corpus)
    tfidf.save("./model/tf_idf.model")
    # tfidf.load("./model/tf_idf.model")


def to_bow(text):
    res = dictionary.doc2bow(tokenize(text))
    return res


def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(tokenize(text))]
    return res


def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1], num_features=len(dictionary))
    sim = index[tfidf2]
    return float(sim[0])


def get_word2vec_vector_bow(text):
    res = np.zeros([100])
    bow = to_bow(text)
    count = 0
    for idx, weight in bow:
        token = dictionary.get(idx)
        if token in word2vec_vocab:
            res += weight * word2vec_vectors[token]
            count += weight
    if count != 0:
        return res / count
    return np.zeros([100])


def get_glove_vector_bow(text):
    res = np.zeros([100])
    text_bow = to_bow(text)
    count = 0
    for idx, weight in text_bow:
        token = dictionary.get(idx)
        if token in glove_vocab:
            res += weight * glove_vectors[token]
            count += weight
    if count != 0:
        return res / count
    return np.zeros([100])


def get_word2vec_vector_tfidf(text):
    res = np.zeros([100])
    text_tfidf = to_tfidf(text)
    count = 0
    for idx, weight in text_tfidf:
        token = dictionary.get(idx)
        if token in word2vec_vocab:
            res += weight * word2vec_vectors[token]
            count += 1
    if count != 0:
        return res
    return np.zeros([100])


def get_glove_vector_tfidf(text):
    res = np.zeros([100])
    text_tfidf = to_tfidf(text)
    count = 0
    for idx, weight in text_tfidf:
        token = dictionary.get(idx)
        if token in glove_vocab:
            res += weight * glove_vectors[token]
            count += 1
    if count != 0:
        return res
    return np.zeros([100])


# def word2vec_bow_cos_sim(w2v1, w2v2):
#     try:
#         # w2v1 = get_word2vec_vector_bow(text1)
#         # w2v2 = get_word2vec_vector_bow(text2)
#         sim = 1 - spatial.distance.cosine(w2v1, w2v2)
#         return float(sim)
#     except:
#         return float(0)
#
#
# def word2vec_tfidf_cos_sim(w2v1, w2v2):
#     try:
#         # w2v1 = get_word2vec_vector_tfidf(text1)
#         # w2v2 = get_word2vec_vector_tfidf(text2)
#         sim = 1 - spatial.distance.cosine(w2v1, w2v2)
#         return float(sim)
#     except:
#         return float(0)
#
#
# def glove_bow_cos_sim(w2v1, w2v2):
#     try:
#         # w2v1 = get_glove_vector_bow(text1)
#         # w2v2 = get_glove_vector_bow(text2)
#         sim = 1 - spatial.distance.cosine(w2v1, w2v2)
#         return float(sim)
#     except:
#         return float(0)
#
#
# def glove_tfidf_cos_sim(w2v1, w2v2):
#     try:
#         # w2v1 = get_glove_vector_tfidf(text1)
#         # w2v2 = get_glove_vector_tfidf(text2)
#         sim = 1 - spatial.distance.cosine(w2v1, w2v2)
#         return float(sim)
#     except:
#         return float(0)


def get_features(df):
    print("# Start time: %s" % (str(datetime.datetime.now())))
    print("  weighted word2vec features")

    df["z_tfidf_cos_sim"] = df.apply(lambda row: cos_sim(row["s1"], row["s2"]), axis=1)

    df["s1_w2v_bow"] = df.s1.map(lambda x: get_word2vec_vector_bow(x))
    df["s2_w2v_bow"] = df.s2.map(lambda x: get_word2vec_vector_bow(x))
    df["z_w2v_bow_dis_cosine"] = df.apply(lambda row: float(spatial.distance.cosine(row["s1_w2v_bow"], row["s2_w2v_bow"])), axis=1)
    df["z_w2v_bow_dis_euclidean"] = df.apply(lambda row: float(spatial.distance.euclidean(row["s1_w2v_bow"], row["s2_w2v_bow"])), axis=1)
    df["z_w2v_bow_dis_minkowski"] = df.apply(lambda row: float(spatial.distance.minkowski(row["s1_w2v_bow"], row["s2_w2v_bow"])), axis=1)
    df["z_w2v_bow_dis_cityblock"] = df.apply(lambda row: float(spatial.distance.cityblock(row["s1_w2v_bow"], row["s2_w2v_bow"])), axis=1)
    df["z_w2v_bow_dis_canberra"] = df.apply(lambda row: float(spatial.distance.canberra(row["s1_w2v_bow"], row["s2_w2v_bow"])), axis=1)

    df["s1_w2v_tfidf"] = df.s1.map(lambda x: get_word2vec_vector_tfidf(x))
    df["s2_w2v_tfidf"] = df.s2.map(lambda x: get_word2vec_vector_tfidf(x))
    df["z_w2v_tfidf_dis_cosine"] = df.apply(lambda row: float(spatial.distance.cosine(row["s1_w2v_tfidf"], row["s2_w2v_tfidf"])), axis=1)
    df["z_w2v_tfidf_dis_euclidean"] = df.apply(lambda row: float(spatial.distance.euclidean(row["s1_w2v_tfidf"], row["s2_w2v_tfidf"])), axis=1)
    df["z_w2v_tfidf_dis_minkowski"] = df.apply(lambda row: float(spatial.distance.minkowski(row["s1_w2v_tfidf"], row["s2_w2v_tfidf"])), axis=1)
    df["z_w2v_tfidf_dis_cityblock"] = df.apply(lambda row: float(spatial.distance.cityblock(row["s1_w2v_tfidf"], row["s2_w2v_tfidf"])), axis=1)
    df["z_w2v_tfidf_dis_canberra"] = df.apply(lambda row: float(spatial.distance.canberra(row["s1_w2v_tfidf"], row["s2_w2v_tfidf"])), axis=1)

    df["s1_glove_bow"] = df.s1.map(lambda x: get_word2vec_vector_bow(x))
    df["s2_glove_bow"] = df.s2.map(lambda x: get_word2vec_vector_bow(x))
    df["z_glove_bow_dis_cosine"] = df.apply(lambda row: float(spatial.distance.cosine(row["s1_glove_bow"], row["s2_glove_bow"])), axis=1)
    df["z_glove_bow_dis_euclidean"] = df.apply(lambda row: float(spatial.distance.euclidean(row["s1_glove_bow"], row["s2_glove_bow"])), axis=1)
    df["z_glove_bow_dis_minkowski"] = df.apply(lambda row: float(spatial.distance.minkowski(row["s1_glove_bow"], row["s2_glove_bow"])), axis=1)
    df["z_glove_bow_dis_cityblock"] = df.apply(lambda row: float(spatial.distance.cityblock(row["s1_glove_bow"], row["s2_glove_bow"])), axis=1)
    df["z_glove_bow_dis_canberra"] = df.apply(lambda row: float(spatial.distance.canberra(row["s1_glove_bow"], row["s2_glove_bow"])), axis=1)

    df["s1_glove_tfidf"] = df.s1.map(lambda x: get_word2vec_vector_tfidf(x))
    df["s2_glove_tfidf"] = df.s2.map(lambda x: get_word2vec_vector_tfidf(x))
    df["z_glove_tfidf_dis_cosine"] = df.apply(lambda row: float(spatial.distance.cosine(row["s1_glove_tfidf"], row["s2_glove_tfidf"])), axis=1)
    df["z_glove_tfidf_dis_euclidean"] = df.apply(lambda row: float(spatial.distance.euclidean(row["s1_glove_tfidf"], row["s2_glove_tfidf"])), axis=1)
    df["z_glove_tfidf_dis_minkowski"] = df.apply(lambda row: float(spatial.distance.minkowski(row["s1_glove_tfidf"], row["s2_glove_tfidf"])), axis=1)
    df["z_glove_tfidf_dis_cityblock"] = df.apply(lambda row: float(spatial.distance.cityblock(row["s1_glove_tfidf"], row["s2_glove_tfidf"])), axis=1)
    df["z_glove_tfidf_dis_canberra"] = df.apply(lambda row: float(spatial.distance.canberra(row["s1_glove_tfidf"], row["s2_glove_tfidf"])), axis=1)

    del df["s1_w2v_bow"]
    del df["s2_w2v_bow"]
    del df["s1_w2v_tfidf"]
    del df["s2_w2v_tfidf"]
    del df["s1_glove_bow"]
    del df["s2_glove_bow"]
    del df["s1_glove_tfidf"]
    del df["s2_glove_tfidf"]
    df.fillna(0.0)

    print("# End time: %s" % (str(datetime.datetime.now())))

    return df


if __name__ == "__main__":

    print(cos_sim("花呗 分期 古天乐", "花呗 分期 查询"))
    print(to_bow("花呗 分期 古天乐"), to_bow("花呗 分期 查询"))
    print(to_tfidf("花呗 分期 古天乐"), to_tfidf("花呗 分期 查询"))
    print(get_word2vec_vector_bow("花呗 分期 古天乐"))
    print(get_word2vec_vector_tfidf("花呗 分期 古天乐"))
    print(get_glove_vector_bow("花呗 分期 古天乐"))
    print(get_glove_vector_tfidf("花呗 分期 古天乐"))

    train = pd.read_csv(config.path_train_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2", "label"])
    train = get_features(train)
    print(train.shape, train.columns)
    col = [c for c in train.columns if c[:1] == "z"]
    print(col)
    train.to_csv(config.path_train_word2vec, index=False, columns=col, encoding="utf-8")

    # test = pd.read_csv(config.path_test_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = get_features(test)
    # col = [c for c in test.columns if c[:1] == "z"]
    # test.to_csv(config.path_test_word2vec, index=False, columns=col, encoding="utf-8")

