# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import os

import pandas as pd
import xgboost as xgb
import gensim
from gensim.models import Doc2Vec

import config
import cut_utils
import string_distance
import string_diff
import n_grams
import word2vec_utils
import doc2vec_infer
import cdssm_for_stack

print(gensim.__version__)
print(xgb.__version__)

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def process(inpath, outpath):
    # TODO : change inpath when upload to web
    path_test_raw = inpath  # config.path_test_raw

    # use jieba tokenizer the raw sentences
    # cut_utils.del_bom(path_test_raw)
    # test = pd.read_csv(path_test_raw, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = cut_utils.cut(test)
    # test.to_csv(config.path_test_cut, sep=str("\t"), index=False, header=False,
    #             columns=["id", "cut_s1", "cut_s2"], encoding="utf-8")
    #
    # # get string distance features
    # test = pd.read_csv(path_test_raw, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = string_distance.get_features(test)
    # col = [c for c in test.columns if c[:1] == "d"]
    # test.to_csv(config.path_test_string_distance, index=False, columns=col, encoding="utf-8")
    #
    # # get string diff features
    # test = pd.read_csv(config.path_test_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = string_diff.get_features(test)
    # col = [c for c in test.columns if c[:1] == "f"]
    # test.to_csv(config.path_test_string_diff, index=False, columns=col, encoding="utf-8")
    #
    # # get n-grams features
    # test = pd.read_csv(config.path_test_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test["sentences"] = test["s1"] + "_split_tag_" + test["s2"]
    # test = n_grams.get_features(test)
    # col = [c for c in test.columns if c[:1] == "f"]
    # test.to_csv(config.path_test_gram_feature, index=False, columns=col, encoding="utf-8")
    #
    # # get word2vec features
    # test = pd.read_csv(config.path_test_cut, sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2"])
    # test = word2vec_utils.get_features(test)
    # col = [c for c in test.columns if c[:1] == "z"]
    # test.to_csv(config.path_test_word2vec, index=False, columns=col, encoding="utf-8")
    #
    # # get doc2vec features
    # model_name = "doc2vec_model4"
    # abs_path = os.getcwd()
    # model_path = os.path.join(abs_path, "model")
    # model_saved_file = os.path.join(model_path, model_name)
    # doc2vec_model = Doc2Vec.load(model_saved_file)
    # test = pd.read_csv(config.path_test_cut, sep="\t", encoding="utf-8", header=None, names=["id", "s1", "s2"])
    # test = doc2vec_infer.make_feature(test, loaded_model=doc2vec_model)
    # col = [c for c in test.columns if c[:1] == "z"]
    # test.to_csv(config.path_test_doc2vec4, index=False, columns=col, encoding="utf-8")

    # columns
    origincol = ["id", "s1", "s2"]
    copycol2 = ['f_1dis', 'f_2word_dis', 'f_2char_dis', 'f_3word_dis', 'f_3char_dis',
                'f_1dis2', 'f_2word_dis2', 'f_2char_dis2', 'f_3word_dis2', 'f_3char_dis2',
                'f_1dis3', 'f_2word_dis3', 'f_2char_dis3', 'f_3word_dis3', 'f_3char_dis3',
                'f_1dis4', 'f_2word_dis4', 'f_2char_dis4', 'f_3word_dis4', 'f_3char_dis4']
    copycol12 = ['z3_cosine', 'z3_manhatton', 'z3_euclidean', 'z3_pearson', 'z3_spearman', 'z3_kendall']
    copycol13 = ['f_total_unique_words', 'f_wc_diff', 'f_wc_ratio', 'f_wc_diff_unique',
                 'f_wc_ratio_unique', 'f_char_diff', 'f_char_ratio']
    copycol18 = ["d_nlevenshtein_1", "d_nlevenshtein_2", "d_jaro_winkler", "d_jaccard"]
    copycol19 = ["z_tfidf_cos_sim",
                 "z_w2v_bow_dis_cosine", "z_w2v_bow_dis_euclidean", "z_w2v_bow_dis_minkowski",
                 "z_w2v_bow_dis_cityblock", "z_w2v_bow_dis_canberra",
                 "z_w2v_tfidf_dis_cosine", "z_w2v_tfidf_dis_euclidean", "z_w2v_tfidf_dis_minkowski",
                 "z_w2v_tfidf_dis_cityblock", "z_w2v_tfidf_dis_canberra",
                 "z_glove_bow_dis_cosine", "z_glove_bow_dis_euclidean", "z_glove_bow_dis_minkowski",
                 "z_glove_bow_dis_cityblock", "z_glove_bow_dis_canberra",
                 "z_glove_tfidf_dis_cosine", "z_glove_tfidf_dis_euclidean", "z_glove_tfidf_dis_minkowski",
                 "z_glove_tfidf_dis_cityblock", "z_glove_tfidf_dis_canberra"]

    test_raw = pd.read_csv(path_test_raw, sep="\t", names=origincol, encoding="utf-8")
    test_feature2 = pd.read_csv(config.path_test_gram_feature, usecols=copycol2, dtype=float, encoding="utf-8")
    test_feature12 = pd.read_csv(config.path_test_doc2vec4, usecols=copycol12, dtype=float, encoding="utf-8")
    test_feature13 = pd.read_csv(config.path_test_string_diff, usecols=copycol13, dtype=float, encoding="utf-8")
    test_feature18 = pd.read_csv(config.path_test_string_distance, usecols=copycol18, dtype=float, encoding="utf-8")
    test_feature19 = pd.read_csv(config.path_test_word2vec, usecols=copycol19, dtype=float, encoding="utf-8")
    test_all = pd.concat([test_feature2, test_feature12, test_feature13, test_feature18, test_feature19], axis=1)
    print(test_all.shape)
    ids = test_raw["id"]
    dtest = xgb.DMatrix(test_all)
    print(dtest.num_col())
    bst = xgb.Booster(model_file="./model/0005.model")  # init model
    ypred = bst.predict(dtest)

    ypred = pd.DataFrame(list(map(lambda x: int(round(x)), ypred)))
    xgb_df = pd.concat([ids, ypred], axis=1)
    xgb_df.to_csv("test_xgb.csv", sep=str("\t"), header=False, index=False, encoding="utf-8")
    xgb_df = pd.read_csv("test_xgb.csv", sep="\t", header=None, names=["idx", "pred1"])
    print(xgb_df.head())


if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])






