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

import os

import pandas as pd
from fasttext import load_model

if __name__ == "__main__":
    # sub = pd.read_csv('../data/提交示例.csv')
    sub = pd.read_csv('../data/data_B/test.csv')
    sub['label'] = 0
    test_data = os.path.join("", 'fasttext.train_fold0')
    for fold in range(5):
        model = load_model(f'cooking{fold}.bin')
        with open(test_data, 'r', encoding='utf-8') as f:
            data = f.readlines()
            data = [line.strip() for line in data]
        all_labels, all_probs = model.predict(data)
        # print(all_probs)
        all_probs = [prob[0] for prob in all_probs]
        # print(sub)
        sub[f'label_{fold}'] = all_probs
    sub['label'] = sub[[f'label_{fold}' for fold in range(5)]].mean(axis=1)
    sub['label'] = 1 - sub['label']
    sub[['pid', 'label']].to_csv('../results/0.80794_fastext.csv', index=None)
