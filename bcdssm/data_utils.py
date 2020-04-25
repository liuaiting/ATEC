# coding=utf-8
from __future__ import print_function

import codecs
import jieba
import os
jieba.load_userdict('userdict.txt')

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
