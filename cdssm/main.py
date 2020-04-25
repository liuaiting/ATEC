# -*- coding: utf-8 -*-
"""
Created on Wed March 29 2018.
@author: Liu Aiting

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import pickle
import codecs
import pdb

import torch
from torchtext import data

import model_cdssm as model
import train_cdssm as train
import data_utils as utils


def process(inpath, outpath):
    parser = argparse.ArgumentParser(description='CDSSM')
    # learning
    parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('-seq-len', type=int, default=50, help='length of the question [default: 20]')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-embed-dim', type=int, default=200, help='number of embedding dimension [default: 200]')
    parser.add_argument('-hidden-dim', type=int, default=256, help='number of hidden dimension [default: 256]')
    parser.add_argument('-num_layers', type=int, default=1, help='number of hidden layers [default: 3]')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

    parser.add_argument('inpath', type=argparse.FileType('r'))
    parser.add_argument('outpath', type=argparse.FileType('w'))

    args = parser.parse_args()

    id_field = data.LabelField(sequential=False, use_vocab=False)
    sentence_field = data.Field(sequential=True, fix_length=args.seq_len)
    sentence_field.vocab = pickle.load(codecs.open('sentence_vocab.pk', 'rb'))

    # Update args and print
    args.vocab_size = len(sentence_field.vocab)
    args.cuda = (not args.no_cuda) and torch.cuda.is_available();del args.no_cuda

    # print("\nParameters:")
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value))

    # model
    cdssm = model.CDSSM(args)
    # TODO: change model file here
    args.snapshot = 'snapshot/2018-06-24_23-06-50/best_steps_12100.pt'

    cdssm.load_state_dict(torch.load(args.snapshot))
    # except Exception:
    #     print("No such file or directory: '{}'".format(args.snapshot))
    # else:
    #     print('\nLoading model from {}...'.format(args.snapshot))
    # finally:
    #     print('Loading model successful.')

    if args.cuda:
        torch.cuda.set_device(args.device)
        cdssm = cdssm.cuda()
    utils.del_bom(inpath)
    utils.cut_test(inpath, 'test_cut.csv')
    # pdb.set_trace()

    train.predict('test_cut.csv', outpath, cdssm, sentence_field, id_field, args.cuda)


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])









