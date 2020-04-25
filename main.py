# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import sys
import os
from collections import Counter

import pandas as pd

import cdssm.main_cdssm_stack as cdssm_stack
import xgb_model.main_xgb_stack as xgb_stack

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def stack(row):
    preds = row.iloc[1:].values.tolist()
    count = Counter()
    count.update(preds)

    score = count.get(1)
    if score and score / len(preds) > 0.5:
        return 1
    else:
        return 0


def process(inpath, outpath):
    # xgb_stack.process(inpath, "test_xgb.csv")
    xgb_df = pd.read_csv("test_xgb.csv", sep="\t", header=None, names=["idx", "pred1"], dtype=int)

    # cdssm_stack.process(inpath, "test_cdssm.csv")
    cdssm_df = pd.read_csv("test_cdssm.csv", sep="\t", header=None, names=["idx", "pred2"], dtype=int)
    print(cdssm_df.head())
    res_df = pd.concat([xgb_df["idx"], xgb_df["pred1"], cdssm_df["pred2"]], axis=1)
    res_df["label"] = res_df.apply(lambda row: stack(row), axis=1)
    del(res_df["pred1"])
    del(res_df["pred2"])
    res_df.to_csv(outpath, sep=str("\t"), encoding="utf-8", index=False, header=False)
    print(res_df.head())


if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])
