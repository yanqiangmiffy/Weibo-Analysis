#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
from fasttext import load_model
import os
from fasttext import train_supervised
from sklearn.metrics import accuracy_score,f1_score,classification_report

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


def eval(valid_data):
    model = load_model(f'cooking{fold}.bin')
    with open(valid_data, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()
        data = ["".join(line.split(' ')[1:]).strip() for line in raw_data]
        true_labels = [line.split(' ')[:1][0] for line in raw_data]
    all_labels, all_probs = model.predict(data)
    all_probs = [prob[0] for prob in all_probs]
    all_labels = [all_labels[0] for label in all_labels]
    print(true_labels)
    # print(all_labels)
    print(accuracy_score(true_labels,all_labels))
    print(f1_score(true_labels,all_labels,average='macro'))
    print(classification_report(true_labels,all_labels))
if __name__ == "__main__":
    for fold in range(5):
        # train_data = os.path.join("../data/2023_A", f'fasttext.train_fold{fold}')
        # valid_data = os.path.join("../data/2023_A", f'fasttext.train_fold{fold}')
        train_data = os.path.join(".", f'fasttext.train_fold{fold}')
        valid_data = os.path.join(".", f'fasttext.train_fold{fold}')
        # train_supervised uses the same arguments and defaults as the fastText cli
        model = train_supervised(
            input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
        )
        print_results(*model.test(valid_data))

        model = train_supervised(
            input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1,
            loss="hs"
        )
        print_results(*model.test(valid_data))
        model.save_model(f"cooking{fold}.bin")
        eval(valid_data)
        # model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
        # print_results(*model.test(valid_data))
        # model.save_model("cooking.ftz")

    # model.predict()
    """
    Number of words:  201149
    Number of labels: 2
    
    Read 7M words
    Number of words:  114643
    Number of labels: 2
    
    Number of words:  109903
    Number of labels: 2
    """