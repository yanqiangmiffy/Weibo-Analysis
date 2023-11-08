from functools import partial
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd


def dosomething(row):
    # 添加计算操作

    return


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


class TopicClassifier():
    def __init__(self):
        self.tfidf_mdoel = joblib.load('models/tfidf_model.joblib')
        self.cls_model = joblib.load('models/classifier.joblib')
        self.le_model = joblib.load('models/le.joblib')

    def process(self, text):
        # text = ' '.join([w for w in jieba.lcut(text)])
        vector = self.tfidf_mdoel.transform([text])
        return vector

    def predict(self, text):
        vector = self.process(text)
        prob = self.cls_model.predict(vector)
        label = self.le_model.inverse_transform(prob)[0]
        return label

# if __name__ == '__main__':  # 不要忘了这句话，否则运行.py文件时，无法启动multiprocessing
#     data['newcol'] = parallelize_on_rows(data, dosomething)
