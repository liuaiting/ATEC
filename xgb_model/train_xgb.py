# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import config

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    origincol = ["id", "s1", "s2", "label"]
    copycol2 = ['f_1dis', 'f_2word_dis', 'f_2char_dis', 'f_3word_dis', 'f_3char_dis',
                'f_1dis2', 'f_2word_dis2', 'f_2char_dis2', 'f_3word_dis2', 'f_3char_dis2',
                'f_1dis3', 'f_2word_dis3', 'f_2char_dis3', 'f_3word_dis3', 'f_3char_dis3',
                'f_1dis4', 'f_2word_dis4', 'f_2char_dis4', 'f_3word_dis4', 'f_3char_dis4']
    copycol12 = ['z3_cosine', 'z3_manhatton', 'z3_euclidean', 'z3_pearson', 'z3_spearman', 'z3_kendall']
    copycol13 = ['f_total_unique_words', 'f_wc_diff', 'f_wc_ratio', 'f_wc_diff_unique',
                 'f_wc_ratio_unique', 'f_char_diff', 'f_char_ratio']
    copycol18 = ["d_nlevenshtein_1", "d_nlevenshtein_2", "d_jaro_winkler", "d_jaccard"]
    copycol19 = ["z_tfidf_cos_sim",
                 "z_w2v_bow_dis_cosine", "z_w2v_bow_dis_euclidean", "z_w2v_bow_dis_minkowski", "z_w2v_bow_dis_cityblock", "z_w2v_bow_dis_canberra",
                 "z_w2v_tfidf_dis_cosine", "z_w2v_tfidf_dis_euclidean", "z_w2v_tfidf_dis_minkowski", "z_w2v_tfidf_dis_cityblock", "z_w2v_tfidf_dis_canberra",
                 "z_glove_bow_dis_cosine", "z_glove_bow_dis_euclidean", "z_glove_bow_dis_minkowski", "z_glove_bow_dis_cityblock", "z_glove_bow_dis_canberra",
                 "z_glove_tfidf_dis_cosine", "z_glove_tfidf_dis_euclidean", "z_glove_tfidf_dis_minkowski", "z_glove_tfidf_dis_cityblock", "z_glove_tfidf_dis_canberra"]

    train_raw = pd.read_csv(config.path_train_raw, sep="\t", names=origincol, encoding="utf-8")
    train_feature2 = pd.read_csv(config.path_train_gram_feature, usecols=copycol2, dtype=float, encoding="utf-8")
    train_feature12 = pd.read_csv(config.path_train_doc2vec4, usecols=copycol12, dtype=float, encoding="utf-8")
    train_feature13 = pd.read_csv(config.path_train_string_diff, usecols=copycol13, dtype=float, encoding="utf-8")
    train_feature18 = pd.read_csv(config.path_train_string_distance, usecols=copycol18, dtype=float, encoding="utf-8")
    train_feature19 = pd.read_csv(config.path_train_word2vec, usecols=copycol19, dtype=float, encoding="utf-8")
    # print(train_feature19.head(5))
    # print(train_feature18.head(5))

    train = train_raw["label"]
    train = pd.concat([train, train_feature2, train_feature12, train_feature13, train_feature18, train_feature19], axis=1)
    print(train.shape)

    train_features = train.iloc[:, 1:train.shape[1]]
    train_labels = train.iloc[:, 0]

    x_train, x_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.1, shuffle=True)
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    # dtest = xgb_model.DMatrix(test, label=None)

    # setting parameters
    param = {
        "booster": "gbtree",
        "silent": 1,
        "nthread": 80,
        "eta": 0.1,
        "gamma": 0.01,
        "max_depth": 6,
        "min_child_weight": 1,
        "max_delta_step": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "colsample_bylevel": 0.5,
        "lambda": 1,
        "alpha": 0,
        "scale_pos_weight": 4,
        "objective": "binary:logistic",
        "eval_metric": 'logloss',
    }
    evallist = [(dval, "eval"), (dtrain, "train")]

    # training
    num_round = 2000
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)

    # save model
    bst.save_model("./model/0005.model")

    # evaluate with f1
    ypred = bst.predict(dval)
    ypred = list(map(round, ypred))
    ytrue = dval.get_label()
    f1 = f1_score(ytrue, ypred)
    print("f1 : %.4f" % f1)

    # # load model, plot importance
    # import matplotlib.pyplot as plt
    # bst = xgb_model.Booster(model_file="./model/0005.model")  # init model
    # xgb_model.plot_importance(bst)
    # plt.savefig("plot_importance")

    # from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import StratifiedKFold
    # model = xgb_model.XGBClassifier(max_depth=3, n_estimators=1000, subsample=0.8, scale_pos_weight=4)
    # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # param_grid = dict(learning_rate=learning_rate)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    # grid_result = grid_search.fit(x_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #
    # means = grid_result.cv_results_["mean_test_score"]
    # stds = grid_result.cv_results_["std_test_score"]
    # params = grid_result.cv_results_["params"]
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":
    main()
