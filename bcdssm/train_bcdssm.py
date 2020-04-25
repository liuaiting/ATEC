from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import time
import codecs

import torch
import torchtext.data as data
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#
#
# def train(train_iter, val_iter, model, args, sentence_field, label_field):
#     start_time = time.time()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     loss_func = torch.nn.CrossEntropyLoss()
#
#     steps = 0
#     best_acc = 0
#     last_step = 0
#     model.train()
#
#     plot_steps = list()
#     plot_losses = list()
#     plot_accs = list()
#     annotate_best_step = 0
#     annotate_best_acc = 0
#
#     for epoch in range(1, args.epochs + 1):
#         for batch in train_iter:
#             sentence1 = batch.sentence1
#             sentence2 = batch.sentence2
#             target = batch.label
#             sentence1 = sentence1.t_()
#             sentence2 = sentence2.t_()
#             # target = target.t_()
#
#             model.batch_size = batch.batch_size
#
#             if args.cuda:
#                 sentence1 = sentence1.cuda()
#                 sentence2 = sentence2.cuda()
#
#             # print(sentence1.data.size())
#             # print(sentence2.data.size())
#             # print(sentence1_lengths.data)
#             # orig_text = sentence_field.reverse(sentence1.data)
#             # print(type(orig_text), len(orig_text))
#             # print(orig_text[0])
#             # print(target.data)
#
#             optimizer.zero_grad()
#             logit = model(sentence1, sentence2)
#             loss = loss_func(logit, target)
#             # print(loss)
#             loss.backward()
#             optimizer.step()
#
#             steps += 1
#             if steps % args.log_interval == 0:
#                 plot_losses.append(loss.data[0].item())
#                 plot_steps.append(steps)
#
#                 target = target.data
#                 logit = torch.max(logit, 1)[1].data
#                 accuracy = accuracy_score(target, logit)
#                 # recall = recall_score(target, logit)
#                 # precision = precision_score(target, logit)
#                 f1 = f1_score(target, logit)
#                 plot_accs.append(accuracy)
#                 sys.stdout.write(
#                     '\rBatch[{}-{}] - loss: {:.6f}  acc: {}  f1: {}  time: {}s'.format(
#                         epoch, steps, loss.data[0].item(), accuracy, f1, time.time()-start_time))
#             if steps % args.test_interval == 0:
#                 dev_acc = eval(val_iter, model, args)
#                 if dev_acc > best_acc:
#                     best_acc = dev_acc
#                     last_step = steps
#                     annotate_best_acc = best_acc
#                     annotate_best_step = last_step
#                     if args.save_best:
#                         save(model, args.save_dir, 'best', steps)
#                 else:
#                     if steps - last_step >= args.early_stop:
#                         print('early stop by {} steps.'.format(args.early_stop))
#             elif steps % args.save_interval == 0:
#                 save(model, args.save_dir, 'snapshot', steps)
#
#         plt.figure()
#         x = np.array(plot_steps)
#         y1 = np.array(plot_losses)
#         y2 = np.array(plot_accs)
#
#         plt.subplot(211)
#         plt.plot(x, y1, 'r-')
#         plt.xlabel('step')
#         plt.ylabel('loss')
#
#         plt.subplot(212)
#         plt.plot(x, y2, 'b-')
#         plt.annotate('best_acc {}'.format(annotate_best_acc), xy=(annotate_best_step, annotate_best_acc),
#                      xytext=(annotate_best_step - 100, annotate_best_acc - 100),
#                      arrowprops=dict(facecolor='black', shrink=0.05), )
#         plt.xlabel('step')
#         plt.ylabel('accuracy')
#
#         plt.savefig('./img/bcdssm_acc_epoch{}.jpg'.format(epoch))


def eval(data_iter, model, args):
    start_time = time.time()
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    logits = []
    targets = []

    corrects, avg_loss = 0, 0
    for batch in data_iter:
        sentence1 = batch.sentence1
        sentence2 = batch.sentence2
        target = batch.label
        sentence1 = sentence1.t_()
        sentence2 = sentence2.t_()

        model.batch_size = batch.batch_size

        if args.cuda:
            sentence1 = sentence1.cuda()
            sentence2 = sentence2.cuda()

        logit = model(sentence1, sentence2)

        loss = loss_func(logit, target)

        avg_loss += loss.data[0]

        logit = torch.max(logit, 1)[1].data.tolist()
        target = target.data.tolist()

        logits.extend(logit)
        targets.extend(target)

    f1 = f1_score(targets, logits)
    accuracy = accuracy_score(targets, logits)

    size = len(data_iter.dataset)
    avg_loss /= size
    print('\nEvaluation - loss: {:.6f}  acc: {}   f1: {}  time: {}s\n'.format(
        avg_loss, accuracy, f1, time.time()-start_time))
    return f1


def predict(inpath, outpath, model, sentence_field, id_field, cuda_flag):
    predict_data = data.TabularDataset(inpath, format='tsv',
                                       fields=[("id", id_field),
                                               ('sentence1', sentence_field),
                                               ('sentence2', sentence_field)])
    batch_size = 64
    # print('DATA_SIZE={}'.format(len(predict_data)))
    # print('BATCH_SIZE={}'.format(batch_size))

    predict_iter = data.Iterator(predict_data, batch_size=batch_size, device=-1, repeat=False, shuffle=False)
    steps = 0
    for batch in predict_iter:
        sentence1 = batch.sentence1
        sentence2 = batch.sentence2
        # orig_sent1 = sentence_field.reverse(sentence1)
        # orig_sent2 = sentence_field.reverse(sentence2)
        sentence1 = sentence1.t_()
        sentence2 = sentence2.t_()
        if cuda_flag:
            sentence1 = sentence1.cuda()
            sentence2 = sentence2.cuda()
        model.batch_size = batch.batch_size

        logit = model(sentence1, sentence2)

        steps += 1
        # sys.stdout.write('\rBatch[{}]'.format(steps))

        idx = batch.id.tolist()
        predict_label = torch.max(logit, 1)[1].data.tolist()
        results = [str(i)+'\t'+str(l)+'\n' for i, l in zip(idx, predict_label)]
        # print(results)
        with codecs.open(outpath, 'a', 'utf-8') as fw:
            fw.writelines(results)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
