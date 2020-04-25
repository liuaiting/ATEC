# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import datetime
import multiprocessing
import os

import pandas as pd
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec

import config


def tokenizer(s):
    return s.strip().split()


def Cosine(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    res = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 0.000001)
    return res


def Manhattan(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    res = np.sum(np.abs(vec1 - vec2))
    # res = np.linalg.norm(vec1 - vec2, ord=1)
    return res


def Euclidean(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    res = np.sqrt(np.sum(np.square(vec1 - vec2)))
    # res = np.linalg.norm(vec1-vec2)
    return res


def PearsonSimilar(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('pearson')[0][1]


def SpearmanSimilar(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('spearman')[0][1]


def KendallSimilar(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('kendall')[0][1]


def get_sentence_vector(sentence, loaded_model):
    tokenize_sentence = tokenizer(sentence)
    infer_vector_of_s = loaded_model.infer_vector(tokenize_sentence)
    return infer_vector_of_s


def make_feature(df_features, loaded_model):
    print("# Start time: %s" % (str(datetime.datetime.now())))
    print('  get sentence vector')
    df_features['doc2vec1'] = df_features.s1.map(lambda x: get_sentence_vector(x, loaded_model))
    df_features['doc2vec2'] = df_features.s2.map(lambda x: get_sentence_vector(x, loaded_model))
    print("# tmp time: %s" % (str(datetime.datetime.now())))
    print('  get six kinds of coefficient about vector')
    df_features['z3_cosine'] = df_features.apply(lambda x: Cosine(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_manhatton'] = df_features.apply(lambda x: Manhattan(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_euclidean'] = df_features.apply(lambda x: Euclidean(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_pearson'] = df_features.apply(lambda x: PearsonSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_spearman'] = df_features.apply(lambda x: SpearmanSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    df_features['z3_kendall'] = df_features.apply(lambda x: KendallSimilar(x['doc2vec1'], x['doc2vec2']), axis=1)
    print("# End time: %s" % (str(datetime.datetime.now())))
    return df_features


if __name__ == "__main__":
    model_name = "doc2vec_model4"
    abs_path = os.getcwd()
    model_path = os.path.join(abs_path, "model")
    model_saved_file = os.path.join(model_path, model_name)
    model = Doc2Vec.load(model_saved_file)

    train = pd.read_csv(config.path_train_cut, sep="\t", encoding="utf-8", header=None,
                        names=["id", "s1", "s2", "label"])
    train = make_feature(train, loaded_model=model)
    col = [c for c in train.columns if c[:1] == "z"]
    train.to_csv(config.path_train_doc2vec4, index=False, columns=col, encoding="utf-8")

    # test = pd.read_csv(config.path_test_cut, sep="\t", encoding="utf-8", header=None,
    #                    names=["id", "s1", "s2"])
    # test = make_feature(test, loaded_model=model)
    # col = [c for c in test.columns if c[:1] == "z"]
    # test.to_csv(config.path_test_doc2vec4, index=False, columns=col, encoding="utf-8")

