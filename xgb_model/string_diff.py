# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import datetime

import pandas as pd
import numpy as np

import xgb_model.config as config


def total_unique_words(row):
    """分词后，词序列长度差"""
    return len(set(row["s1"]).union(row["s2"]))


def wc_diff(row):
    """字符串长度差"""
    return abs(len(row["s1"]) - len(row["s2"]))


def wc_ratio(row):
    """字符串长度比"""
    l1 = len(row["s1"]) * 1.0
    l2 = len(row["s2"])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    """词序列长度差"""
    return abs(len(set(row["s1"])) - len(set(row["s2"])))


def wc_ratio_unique(row):
    """词序列长度比"""
    l1 = len(set(row["s1"])) * 1.0
    l2 = len(set(row["s2"]))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff(row):
    """字序列长度差"""
    return abs(len("".join(row["s1"])) - len("".join(row["s2"])))


def char_ratio(row):
    """字序列长度比"""
    l1 = len(''.join(row['s1']))
    l2 = len(''.join(row['s2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def get_features(df_features):
    print("# Start time: %s" % str(datetime.datetime.now()))
    print("  string diff features")
    # compute distance
    df_features["f_total_unique_words"] = df_features.apply(total_unique_words, axis=1)
    df_features["f_wc_diff"] = df_features.apply(wc_diff, axis=1)
    df_features["f_wc_ratio"] = df_features.apply(wc_ratio, axis=1)
    df_features["f_wc_diff_unique"] = df_features.apply(wc_diff_unique, axis=1)
    df_features["f_wc_ratio_unique"] = df_features.apply(wc_ratio_unique, axis=1)
    df_features["f_char_diff"] = df_features.apply(char_diff, axis=1)
    df_features["f_char_ratio"] = df_features.apply(char_ratio, axis=1)

    print("# End time: %s" % str(datetime.datetime.now()))
    return df_features


if __name__ == "__main__":

    train = pd.read_csv(config.path_train_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2", "label"])
    train = get_features(train)
    col = [c for c in train.columns if c[:1] == "f"]
    train.to_csv(config.path_train_string_diff, index=False, columns=col, encoding="utf-8")

    # test = pd.read_csv(config.path_test_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = get_features(test)
    # test.to_csv(config.path_test_string_diff, index=False, columns=col, encoding="utf-8")

    # row = {"s1": "我要 申请 借呗 什么", "s2": "我 开通 不了 借呗"}
    # print(wc_diff(row))
    # print(wc_diff_unique(row))
    # print(wc_ratio(row))
    # print(wc_ratio_unique(row))
    # print(char_diff(row))
    # print(char_ratio(row))


