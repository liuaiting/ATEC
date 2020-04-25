# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import datetime

import pandas as pd
import numpy as np

import xgb_model.config as config


__all__ = ["levenshtein", "nlevenshtein", "jaccard", "jaro_winkler"]

# TODO: 不能使用直接计算字符串距离的工具包，需要自己实现。


def levenshtein(seq1, seq2, max_dist=-1, normalized=False):
    """Compute the absolute Levenshtein distance between the two sequences
    `seq1` and `seq2`."""
    if normalized:
        return nlevenshtein(seq1, seq2, method=1)

    if seq1 == seq2:
        return 0

    len1, len2 = len(seq1), len(seq2)
    if max_dist >= 0 and abs(len1 - len2) > max_dist:
        return -1
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    if len1 < len2:
        len1, len2 = len2, len1
        seq1, seq2 = seq2, seq1

    column = np.array(range(len2 + 1))

    for x in range(1, len1 + 1):
        column[0] = x
        last = x - 1
        for y in range(1, len2 + 1):
            old = column[y]
            cost = int(seq1[x - 1] != seq2[y - 1])
            column[y] = min(column[y] + 1, column[y - 1] + 1, last + cost)
            last = old
        if max_dist >= 0 and min(column) > max_dist:
            return -1

    if max_dist >= 0 and column[len2] > max_dist:
        # stay consistent, even if we have the exact distance
        return -1
    return float(column[len2])


def nlevenshtein(seq1, seq2, method=1):
    """Compute the normalized Levenshtein distance between `seq1` and `seq2`."""

    if seq1 == seq2:
        return 0.0
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 or len2 == 0:
        return 1.0
    if len1 < len2:  # minimize the arrays size
        len1, len2 = len2, len1
        seq1, seq2 = seq2, seq1

    if method == 1:
        return levenshtein(seq1, seq2) / float(len1)
    if method != 2:
        raise ValueError("expected either 1 or 2 for `method` parameter")

    column = np.array(range(len2 + 1))
    length = np.array(range(len2 + 1))

    for x in range(1, len1 + 1):

        column[0] = length[0] = x
        last = llast = x - 1

        for y in range(1, len2 + 1):
            # dist
            old = column[y]
            ic = column[y - 1] + 1
            dc = column[y] + 1
            rc = last + (seq1[x - 1] != seq2[y - 1])
            column[y] = min(ic, dc, rc)
            last = old

            # length
            lold = length[y]
            lic = length[y - 1] + 1 if ic == column[y] else 0
            ldc = length[y] + 1 if dc == column[y] else 0
            lrc = llast + 1 if rc == column[y] else 0
            length[y] = max(ldc, lic, lrc)
            llast = lold

    return column[y] / float(length[y])


def jaccard(seq1, seq2):
    """Compute the Jaccard distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.

    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(seq1), set(seq2)
    return 1 - len(set1 & set2) / float(len(set1 | set2))


def jaro_winkler(string1, string2, prefix_weight=0.1, winklerize=True):
    """Compute Jaro string similarity metric of two strings."""
    s1, s2 = string1, string2
    s1_len = len(string1)
    s2_len = len(string2)

    if not s1_len or not s2_len:
        return 0.0

    min_len = max(s1_len, s2_len)
    search_range = (min_len / 2) - 1
    if search_range < 0:
        search_range = 0

    s1_flags = [False] * s1_len
    s2_flags = [False] * s2_len

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    for i, s1_ch in enumerate(s1):
        low = int(max(0, i - search_range))
        hi = int(min(i + search_range + 1, s2_len))
        for j in range(low, hi):
            if not s2_flags[j] and s2[j] == s1_ch:
                s1_flags[i] = s2_flags[j] = True
                common_chars += 1
                break
    # short circuit if no characters match
    if not common_chars:
        return 0.0

    # count transpositions
    k = trans_count = 0
    for i, s1_f in enumerate(s1_flags):
        if s1_f:
            for j in range(k, s2_len):
                if s2_flags[j]:
                    k = j + 1
                    break
            if s1[i] != s2[j]:
                trans_count += 1
    trans_count /= 2

    # adjust for similarities in nonmatched characters.
    common_chars = float(common_chars)
    weight = common_chars / s1_len + common_chars / s2_len
    weight += (common_chars - trans_count) / common_chars
    weight = weight / 3

    # stop to boost if strings are not similar
    if not winklerize:
        return weight
    if weight <= 0.7 or s1_len <= 3 or s2_len <= 3:
        return weight

    # winkler modification
    # adjust for up to first 4 chars in common
    j = min(min_len, 4)
    i = 0
    while i < j and s1[i] == s2[i] and s1[i]:
        i += 1
    if i:
        weight += i * prefix_weight * (1.0 - weight)

    return weight


def get_features(df_features):
    """
    Read raw csv file, format "id\tsentence1\tsentence2\tlabel" for train/dev dataset,
    format "id\tsentence1\tsentence2" for test dataset.
    Compute distance bewteen sentence1 and sentence2.
    Save the ground truth label and all features in a new csv file.
    # Note : train and dev dataset has four columns, test dataset has three columns(expect "label")
    """
    print("# Start time: %s" % str(datetime.datetime.now()))
    print("  string distance features")
    # compute distance
    df_features["d_nlevenshtein_1"] = df_features.apply(
        lambda row: nlevenshtein(row["s1"], row["s2"], method=1), axis=1)
    df_features["d_nlevenshtein_2"] = df_features.apply(
        lambda row: nlevenshtein(row["s1"], row["s2"], method=2), axis=1)
    df_features["d_jaro_winkler"] = df_features.apply(
        lambda row: jaro_winkler(row["s1"], row["s2"]), axis=1)
    df_features["d_jaccard"] = df_features.apply(
        lambda row: jaccard(row["s1"], row["s2"]), axis=1)

    print("# End time: %s" % str(datetime.datetime.now()))
    return df_features


if __name__ == "__main__":
    pass
    # print(levenshtein("我要 申请 借呗 什么", "我 开通 不了 借呗"))
    # train = pd.read_csv(config.path_train_raw, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2", "label"])
    # train = get_features(train)
    # col = [c for c in train.columns if c[:1] == "d"]
    # train.to_csv(config.path_train_string_distance, index=False, columns=col, encoding="utf-8")

    # test = pd.read_csv(config.path_test_raw, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = get_features(test)
    # test.to_csv(config.path_test_string_distance, index=False, columns=col, encoding="utf-8")
