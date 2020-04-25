# encoding=utf-8
import multiprocessing
import datetime
import os

import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import config


def tokenizer(sentence):
    return sentence.strip().split(" ")


def get_documents(inpath):
    df_train = pd.read_csv(inpath, header=None, names=["idx", "s1", "s2", "label"], sep="\t", encoding="utf-8")
    sentences = []
    for index, record in df_train.iterrows():
        idx = record["idx"]
        s1 = record["s1"]
        s2 = record["s2"]
        sentences.append(TaggedDocument(words=tokenizer(s1), tags=["s1_%s" % idx]))
        sentences.append(TaggedDocument(words=tokenizer(s2), tags=["s2_%s" % idx]))
    return sentences


def train_and_save_doc2vec_model(documents, model="model4", m_epochs=100, m_min_count=0, m_vector_size=100, m_window=5):
    print("# Start time: %s" % str(datetime.datetime.now()))
    cores = multiprocessing.cpu_count()
    model_path = config.path_model
    saved_model_name = "doc2vec_%s" % model
    doc_vec_file = os.path.join(model_path, "%s" % saved_model_name)
    if model == "model1":
        # PV-DBOW
        model_1 = Doc2Vec(documents, dm=0, workers=cores, vector_size=m_vector_size, window=m_window,
                          min_count=m_min_count, epochs=m_epochs, dbow_words=1)
        model_1.save("%s" % doc_vec_file)
        print("# model training completed: %s" % doc_vec_file)
    elif model == "model2":
        # PV-DBOW
        model_2 = Doc2Vec(documents, dm=0, workers=cores, vector_size=m_vector_size, window=m_window,
                          min_count=m_min_count, epochs=m_epochs, dbow_words=0)
        model_2.save("%s" % doc_vec_file)
        print("# model training completed: %s" % doc_vec_file)
    elif model == "model3":
        # PV-DM w/average
        model_3 = Doc2Vec(documents, dm=1, dm_mean=1, workers=cores, vector_size=m_vector_size, window=m_window,
                          min_count=m_min_count, epochs=m_epochs, dbow_words=0)
        model_3.save("%s" % doc_vec_file)
        print("# model training completed: %s" % doc_vec_file)
    elif model == "model4":
        # PV-DM w/concatenation -window=5 (both sides) approximates paper's 10-word total window size
        model_4 = Doc2Vec(documents, dm=1, dm_concat=1, workers=cores, vector_size=m_vector_size, window=m_window,
                          min_count=m_min_count, epochs=m_epochs, dbow_words=0)
        model_4.save("%s" % doc_vec_file)
        print("# model training completed: %s" % doc_vec_file)
    print("# Record count %s" % len(documents))
    print("# End Time: %s" % str(datetime.datetime.now()))


def get_sentence_vector(sentence):
    print("# sentence - %s" % sentence)
    model_name = "doc2vec_model4"
    model_path = config.path_model
    model_saved_file = os.path.join(model_path, model_name)
    model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)
    print("# Model: %s" % model_name)

    tokenize_sentence = tokenizer(sentence)
    infer_vector_of_s = model.infer_vector(tokenize_sentence)
    print("# tokenize_sentence: %s\n# infer_vector_of_s: %s" % (tokenize_sentence, infer_vector_of_s))


if __name__ == "__main__":
    documents = get_documents(config.path_train_cut)
    # train_and_save_doc2vec_model(documents, model="model1")
    # train_and_save_doc2vec_model(documents, model="model2")
    # train_and_save_doc2vec_model(documents, model="model3")
    train_and_save_doc2vec_model(documents, model="model4")

    # s1 = "我 开通 不了 借呗"
    # s2 = "我要 申请 借呗"
    # get_sentence_vector(s1)














