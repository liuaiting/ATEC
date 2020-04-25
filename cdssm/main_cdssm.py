# -*- coding: utf-8 -*-
"""
Created on Wed March 29 2018.
@author: Liu Aiting

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import argparse
import datetime
import pickle
import codecs

import torch
import torchtext.data as data

import model_cdssm as model
import train_cdssm as train
import data_utils as utils


parser = argparse.ArgumentParser(description='CDSSM')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-seq-len', type=int, default=50, help='length of the question [default: 50]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=200, help='number of embedding dimension [default: 200]')
parser.add_argument('-kernel-nums', type=str, default='100,100,100', help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='1,2,3',
                    help='comma-separated kernel size to use for convolution')

# parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', action='store_true', default=False, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

# utils.cut_train()

print('\nLoading data...')
id_field = data.LabelField(sequential=False, use_vocab=False)
label_field = data.LabelField(sequential=False, use_vocab=False)

# def tokenizer(sentence):
#     seg_list = jieba.cut(sentence)
#     return list(seg_list)
# sentence_field = data.ReversibleField(sequential=True, tokenize=tokenizer, fix_length=args.seq_len)
# train_data, val_data = data.TabularDataset.splits(path='./data',
#                                                   train='atec_nlp_sim_train.csv',
#                                                   validation='atec_nlp_sim_train.csv',
#                                                   format='tsv',
#                                                   fields=[("id", id_field),
#                                                           ('sentence1', sentence_field),
#                                                           ('sentence2', sentence_field),
#                                                           ('label', label_field)])

sentence_field = data.Field(sequential=True, fix_length=args.seq_len)
train_data, val_data = data.TabularDataset.splits(path='./data',
                                                  train='train_cut_sample.csv',
                                                  validation='train_cut_sample.csv',
                                                  format='tsv',
                                                  fields=[("id", id_field),
                                                          ('sentence1', sentence_field),
                                                          ('sentence2', sentence_field),
                                                          ('label', label_field)])

sentence_field.build_vocab(train_data)
pickle.dump(sentence_field.vocab, codecs.open('sentence_vocab.pk', 'wb'))
train_iter, val_iter = data.Iterator.splits((train_data, val_data),
                                            batch_sizes=(args.batch_size, args.batch_size),
                                            shuffle=args.shuffle,
                                            device=args.device,
                                            repeat=False)

# Update args and print
args.vocab_size = len(sentence_field.vocab)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
# args.kernel_num = [int(k) for k in args.kernel_sizes.split(',')]
# args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cdssm = model.CDSSM(args)
print(cdssm)
# args.snapshot = 'snapshot/2018-05-23_09-20-33/best_steps_12.pt'
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cdssm.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cdssm = cdssm.cuda()

# train or predict

if args.predict:
    path = 'test.tsv'  # TODO: 预测文件的路径名
    score = train.predict(path, cdssm, sentence_field, label_field, id_field, args.cuda)
elif args.test:
    try:
        train.eval(train_iter, cdssm, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, val_iter, cdssm, args, sentence_field, label_field)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
