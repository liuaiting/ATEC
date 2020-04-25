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

import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(1)


class BCDSSM(nn.Module):
    def __init__(self, args):
        super(BCDSSM, self).__init__()
        self.args = args
        vocab_size = args.vocab_size
        embed_dim = args.embed_dim  # default: 200
        # lamda = 5
        self.batch_size = args.batch_size
        self.hidden_dim = hidden_dim = args.hidden_dim
        self.num_layers = num_layers = args.num_layers
        self.seq_len = args.seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.max_pooling = nn.MaxPool1d(kernel_size=self.seq_len)
        self.dense = nn.Linear(hidden_dim, 128)

        self.cos = nn.CosineSimilarity(dim=1)
        self.fc = nn.Linear(128 * 2, 2)
        self.sigmoid = nn.Sigmoid()

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim // 2)),
                Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim // 2)))

    def get_lstm_features(self, batch_sentence):
        """

        :param batch_sentence: [batch_size, seq_len]
        :return: batch_feats
        """
        self.hidden = self.init_hidden()
        embeds = self.embed(batch_sentence)  # (None, 20, 200)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # (None, 20, 200)
        lstm_feats = self.tanh(lstm_out)
        # print(lstm_feats.size())
        return lstm_feats

    def cnn(self, batch_feats):
        cnn_feats = torch.transpose(self.conv(torch.transpose(batch_feats, 1, 2)), 1, 2)
        cnn_feats = torch.transpose(self.max_pooling(torch.transpose(cnn_feats, 1, 2)), 1, 2)
        cnn_out = self.dense(cnn_feats).squeeze(1)
        # print(cnn_out.data.size())
        return cnn_out

    def sim_score(self, batch_feats1, batch_feats2):
        sim1 = self.cos(batch_feats1, batch_feats2) * 0.5 + 0.5
        sim0 = torch.ones(self.batch_size) - sim1
        sim = torch.cat((sim0.unsqueeze(-1), sim1.unsqueeze(-1)), 1)
        # print(sim.data)
        # print(sim.data.size())
        return sim

    def fc_sigmoid(self, batch_feats1, batch_feats2):
        batch_feats = torch.cat([batch_feats1, batch_feats2], 1)
        batch_feats = self.fc(batch_feats)
        logits = self.sigmoid(batch_feats)
        return logits

    def forward(self, batch_sentence1, batch_sentence2):
        batch_feats1 = self.get_lstm_features(batch_sentence1)
        batch_feats2 = self.get_lstm_features(batch_sentence2)
        batch_feats1 = self.cnn(batch_feats1)
        batch_feats2 = self.cnn(batch_feats2)
        sim_score = self.sim_score(batch_feats1, batch_feats2)
        # print(sim_score.data.size())
        # print(sim_score.data)
        # logits = self.fc_sigmoid(batch_feats1, batch_feats2)
        return sim_score

