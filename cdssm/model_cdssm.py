"""
Created on Wed April 11 2018
@author: Liu Aiting
@email: liuaiting37@gmail.com

Evaluate the similarity between a predicate and a question pattern,
that is the question without the entity mention.
Learn the latent semantic representation of the question.

Flowing paper "Knowledge Base Question Answering Based on Deep Learning Models"(79.57)
Flowing paper "Topic enhanced deep structured semantic models for knowledge base question answering"(82.43)
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)


class CDSSM(nn.Module):
    def __init__(self, args):
        super(CDSSM, self).__init__()
        self.args = args
        vocab_size = args.vocab_size
        embed_dim = args.embed_dim  # default: 200
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.mode = None
        # kernel_nums = args.kernel_nums
        # kernel_sizes = args.kernel_sizes

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # layers for sequence1
        self.convs1 = nn.ModuleList([nn.Conv1d(embed_dim, 100, kernel_size=2),
                                     nn.Conv1d(embed_dim, 100, kernel_size=3),
                                     nn.Conv1d(embed_dim, 100, kernel_size=4)])
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense1 = nn.Linear(300, 128)

        # layers for sequence2
        self.convs2 = nn.ModuleList([nn.Conv1d(embed_dim, 100, kernel_size=2),
                                     nn.Conv1d(embed_dim, 100, kernel_size=3),
                                     nn.Conv1d(embed_dim, 100, kernel_size=4)])
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(300, 128)

        # connect sent1 and sent2
        self.cos = nn.CosineSimilarity(dim=1)

        # self.fc = nn.Linear(256, 2)
        # self.softmax = nn.Softmax(dim=1)

    def sim_score(self, batch_sent1, batch_sent2):
        sim1 = self.cos(batch_sent1, batch_sent2) * 0.5 + 0.5
        sim0 = torch.ones(self.batch_size) - sim1
        logits = torch.cat((sim0.unsqueeze(-1), sim1.unsqueeze(-1)), 1)
        # print(logits.data)
        # print(logits.data.size())
        return logits

    def forward(self, batch_sentence1, batch_sentence2):
        embed_sent1 = self.embed(batch_sentence1)  # (None, 50, 200)
        sent1s = []
        for conv in self.convs1:
            sent1 = conv(torch.transpose(embed_sent1, 1, 2))
            sent1 = F.relu(sent1)
            sent1 = F.max_pool1d(sent1, kernel_size=sent1.shape[2])
            sent1 = sent1.squeeze(2)
            sent1s.append(sent1)
        sent1 = torch.cat(sent1s, 1)
        if self.mode == "train":
            sent1 = self.dropout1(sent1)
        sent1 = self.dense1(sent1)
        # print(sent1.shape)

        embed_sent2 = self.embed(batch_sentence2)  # (None, 50, 200)
        sent2s = []
        for conv in self.convs2:
            sent2 = conv(torch.transpose(embed_sent2, 1, 2))
            sent2 = F.relu(sent2)
            sent2 = F.max_pool1d(sent2, kernel_size=sent2.shape[2])
            sent2 = sent2.squeeze(2)
            sent2s.append(sent2)
        sent2 = torch.cat(sent2s, 1)
        if self.mode == "train":
            sent2 = self.dropout2(sent2)
        sent2 = self.dense2(sent2)
        # print(sent2.shape)

        logits = self.sim_score(sent1, sent2)
        # sent1_sent2 = torch.cat([sent1, sent2], 1)
        # logits = self.softmax(self.fc(sent1_sent2))
        # print(logits)

        return logits
