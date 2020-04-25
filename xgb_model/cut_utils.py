# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import datetime

import pandas as pd
import jieba

import xgb_model.config as config

jieba.load_userdict('xgb_model/userdict.txt')


def del_bom(inpath):
    command = r"sed -i 's/\xEF\xBB\xBF//g' " + inpath
    # print(command)
    os.system(command)


def cut_sentence(sentence):
    return " ".join(list(jieba.cut(sentence)))


def cut(df):
    print("# Start time: %s" % str(datetime.datetime.now()))
    df["cut_s1"] = df.s1.map(cut_sentence)
    df["cut_s2"] = df.s2.map(cut_sentence)
    print("# End time: %s" % str(datetime.datetime.now()))

    return df


if __name__ == "__main__":
    del_bom(config.path_train_raw)
    train = pd.read_csv(config.path_train_raw, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2", "label"])
    train = cut(train)
    train.to_csv(config.path_train_cut, sep=str("\t"), index=False, header=False,
                 columns=["id", "cut_s1", "cut_s2", "label"], encoding="utf-8")

    # del_bom(config.path_test_raw)
    # test = pd.read_csv(config.path_test_raw, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = cut(test)
    # test.to_csv(config.path_test_cut, sep=str("\t"), index=False, header=False,
    #             columns=["id", "cut_s1", "cut_s2"], encoding="utf-8")
