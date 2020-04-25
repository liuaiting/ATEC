# coding=utf-8
from __future__ import print_function

import codecs
import os

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split

jieba.load_userdict('cdssm/userdict.txt')

__all__ = ["cut_train", "cut_test", "del_bom"]


def cut_train(inpath='../data/atec_nlp_sim_train_all.csv', outpath='../data/train_cut.csv'):
    with codecs.open(inpath, 'r', 'utf-8') as f:
        with codecs.open(outpath, 'w', 'utf-8') as fw:
            for line in f:
                line = line.strip().lower().split('\t')
                idx = line[0]
                sent1 = ' '.join(list(jieba.cut(line[1])))
                sent2 = ' '.join(list(jieba.cut(line[2])))
                label = line[3]
                res = '\t'.join([idx, sent1, sent2, label])
                fw.write(res + '\n')


def cut_test(inpath, outpath):
    with codecs.open(inpath, 'r', 'utf-8') as f:
        with codecs.open(outpath, 'w', 'utf-8') as fw:
            for line in f:
                line = line.strip().lower().split('\t')
                idx = line[0]
                sent1 = ' '.join(list(jieba.cut(line[1])))
                sent2 = ' '.join(list(jieba.cut(line[2])))
                res = '\t'.join([idx, sent1, sent2])
                fw.write(res + '\n')


def del_bom(inpath):
    command = r"sed -i 's/\xEF\xBB\xBF//g' " + inpath
    # print(command)
    os.system(command)


def split_neg_pos(data_path="./data/train_cut.csv"):
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8", header=None,
                     names=["idx", "s1", "s2", "label"])
    group_df = df.groupby(by="label")
    print(group_df.size())
    new_df = df.sample(frac=1).reset_index(drop=True)
    print(new_df.head())
    neg_df = new_df[new_df["label"] == 0]
    pos_df = new_df[new_df["label"] == 1]
    neg_df_sample = neg_df.sample(frac=0.2)

    save_df = pd.concat([neg_df_sample, pos_df], axis=0)
    save_df = save_df.sample(frac=1)
    print(save_df.head())
    save_df.to_csv("./data/train_cut_sample.csv", sep="\t",
                   index=False, header=False, encoding="utf-8")


# split_neg_pos()
