# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import datetime
from multiprocessing import Pool

import pandas as pd
import numpy as np

import config
from simhash import Simhash


def tokenizer(sentence):
    return sentence.strip().split()


def Jaccarc(ws1, ws2):
    if isinstance(ws1, str):
        ws1 = ws1.split()
        ws2 = ws2.split()

    tot = len(ws1) + len(ws2) + 1
    same = 0
    for w1 in ws1:
        for w2 in ws2:
            if w1 == w2:
                same += 1
    return float(same) / (float(tot - same) + 0.000001)


def Dice(ws1, ws2):
    if isinstance(ws1, str):
        ws1 = ws1.split()
        ws2 = ws2.split()

    tot = len(ws1) + len(ws2) + 1
    same = 0
    for w1 in ws1:
        for w2 in ws2:
            if w1 == w2:
                same += 1
    return float(same) / (np.sqrt(float(tot)) + 0.000001)


def Ochiai(ws1, ws2):
    if isinstance(ws1, str):
        ws1 = ws1.split()
        ws2 = ws2.split()

    tot = len(ws1) * len(ws2) + 1
    same = 0
    for w1 in ws1:
        for w2 in ws2:
            if w1 == w2:
                same += 1
    return float(same) / (np.sqrt(float(tot)) + 0.000001)


def calculate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))


# n-grams
def ngrams(sequence, n):
    ngrams_list = []
    for num in range(0, len(sequence) - n + 1):
        ngram = ' '.join(sequence[num: num + n])
        ngrams_list.append(ngram)
    return ngrams_list


def get_word_ngrams(sequence, n=3):
    tokens = tokenizer(sequence)
    return ngrams(tokens, n)


def get_char_ngrams(sequence, n=3):
    sequence = "".join(sequence.split())
    return [sequence[i:i+n] for i in range(len(sequence) - n + 1)]


def get_word_distance(row):
    s1, s2 = row.split("_split_tag_")
    s1, s2 = tokenizer(s1), tokenizer(s2)
    return calculate_simhash_distance(s1, s2)


def get_word_2gram_distance(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 2)
    s2 = get_word_ngrams(s2, 2)
    return calculate_simhash_distance(s1, s2)


def get_char_2gram_distance(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 2)
    s2 = get_char_ngrams(s2, 2)
    return calculate_simhash_distance(s1, s2)


def get_word_3gram_distance(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 3)
    s2 = get_word_ngrams(s2, 3)
    return calculate_simhash_distance(s1, s2)


def get_char_3gram_distance(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 3)
    s2 = get_char_ngrams(s2, 3)
    return calculate_simhash_distance(s1, s2)


def get_word_distance2(row):
    s1, s2 = row.split("_split_tag_")
    s1, s2 = tokenizer(s1), tokenizer(s2)
    return Jaccarc(s1, s2)


def get_word_2gram_distance2(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 2)
    s2 = get_word_ngrams(s2, 2)
    return Jaccarc(s1, s2)


def get_char_2gram_distance2(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 2)
    s2 = get_char_ngrams(s2, 2)
    return Jaccarc(s1, s2)


def get_word_3gram_distance2(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 3)
    s2 = get_word_ngrams(s2, 3)
    return Jaccarc(s1, s2)


def get_char_3gram_distance2(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 3)
    s2 = get_char_ngrams(s2, 3)
    return Jaccarc(s1, s2)


def get_word_distance3(row):
    s1, s2 = row.split("_split_tag_")
    s1, s2 = tokenizer(s1), tokenizer(s2)
    return Dice(s1, s2)


def get_word_2gram_distance3(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 2)
    s2 = get_word_ngrams(s2, 2)
    return Dice(s1, s2)


def get_char_2gram_distance3(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 2)
    s2 = get_char_ngrams(s2, 2)
    return Dice(s1, s2)


def get_word_3gram_distance3(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 3)
    s2 = get_word_ngrams(s2, 3)
    return Dice(s1, s2)


def get_char_3gram_distance3(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 3)
    s2 = get_char_ngrams(s2, 3)
    return Dice(s1, s2)


def get_word_distance4(row):
    s1, s2 = row.split("_split_tag_")
    s1, s2 = tokenizer(s1), tokenizer(s2)
    return Ochiai(s1, s2)


def get_word_2gram_distance4(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 2)
    s2 = get_word_ngrams(s2, 2)
    return Ochiai(s1, s2)


def get_char_2gram_distance4(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 2)
    s2 = get_char_ngrams(s2, 2)
    return Ochiai(s1, s2)


def get_word_3gram_distance4(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_word_ngrams(s1, 3)
    s2 = get_word_ngrams(s2, 3)
    return Ochiai(s1, s2)


def get_char_3gram_distance4(row):
    s1, s2 = row.split("_split_tag_")
    s1 = get_char_ngrams(s1, 3)
    s2 = get_char_ngrams(s2, 3)
    return Ochiai(s1, s2)


def get_features(df_features):
    pool = Pool(80)
    print("# Start time: %s" % str(datetime.datetime.now()))
    print('  n-grams features')
    df_features['f_1dis'] = pool.map(get_word_distance, df_features['sentences'])
    df_features['f_2word_dis'] = pool.map(get_word_2gram_distance, df_features['sentences'])
    df_features['f_2char_dis'] = pool.map(get_char_2gram_distance, df_features['sentences'])
    df_features['f_3word_dis'] = pool.map(get_word_3gram_distance, df_features['sentences'])
    df_features['f_3char_dis'] = pool.map(get_char_3gram_distance, df_features['sentences'])

    df_features['f_1dis2'] = pool.map(get_word_distance2, df_features['sentences'])
    df_features['f_2word_dis2'] = pool.map(get_word_2gram_distance2, df_features['sentences'])
    df_features['f_2char_dis2'] = pool.map(get_char_2gram_distance2, df_features['sentences'])
    df_features['f_3word_dis2'] = pool.map(get_word_3gram_distance2, df_features['sentences'])
    df_features['f_3char_dis2'] = pool.map(get_char_3gram_distance2, df_features['sentences'])

    df_features['f_1dis3'] = pool.map(get_word_distance3, df_features['sentences'])
    df_features['f_2word_dis3'] = pool.map(get_word_2gram_distance3, df_features['sentences'])
    df_features['f_2char_dis3'] = pool.map(get_char_2gram_distance3, df_features['sentences'])
    df_features['f_3word_dis3'] = pool.map(get_word_3gram_distance3, df_features['sentences'])
    df_features['f_3char_dis3'] = pool.map(get_char_3gram_distance3, df_features['sentences'])

    df_features['f_1dis4'] = pool.map(get_word_distance4, df_features['sentences'])
    df_features['f_2word_dis4'] = pool.map(get_word_2gram_distance4, df_features['sentences'])
    df_features['f_2char_dis4'] = pool.map(get_char_2gram_distance4, df_features['sentences'])
    df_features['f_3word_dis4'] = pool.map(get_word_3gram_distance4, df_features['sentences'])
    df_features['f_3char_dis4'] = pool.map(get_char_3gram_distance4, df_features['sentences'])

    print("# End time: %s" % str(datetime.datetime.now()))
    df_features.fillna(0.0)
    return df_features


if __name__ == "__main__":

    train = pd.read_csv(config.path_train_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2", "label"])
    train["sentences"] = train["s1"] + "_split_tag_" + train["s2"]
    train = get_features(train)
    col = [c for c in train.columns if c[:1] == "f"]
    train.to_csv(config.path_train_gram_feature, index=False, columns=col, encoding="utf-8")

    # test = pd.read_csv(config.path_test_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test["sentences"] = test["s1"] + "_split_tag_" + test["s2"]
    # test = get_features(test)
    # test.to_csv(config.path_test_gram_feature, index=False, columns=col, encoding="utf-8")

    # row = {"s1": "我要 申请 借呗 什么", "s2": "我 开通 不了 借呗"}
    # print(wc_diff(row))
    # print(wc_diff_unique(row))
    # print(wc_ratio(row))
    # print(wc_ratio_unique(row))
    # print(char_diff(row))
    # print(char_ratio(row))
    # s1 = "我要 申请 借呗 什么"
    # print(get_char_ngrams(s1))
    # print(get_word_ngrams(s1))
    # print(get_char_2gram_distance2("我要 申请 借呗 什么_split_tag_我 开通 不了 借呗"))







