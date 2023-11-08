import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from pprint import pprint
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import re
import jieba
from sklearn.linear_model import LogisticRegression
import joblib
def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    return contents, labels
base_dir = 'data/thcnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')

train_contents, train_labels = read_file(train_dir)
test_contents, test_labels = read_file(test_dir)

print(train_contents[:10])

#去除文本中的表情字符（只保留中英文和数字）
def clear_character(sentence):
    pattern1= '\[.*?\]'
    pattern2 = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line1=re.sub(pattern1,'',sentence)
    line2=re.sub(pattern2,'',line1)
    new_sentence=''.join(line2.split()) #去除空白
    return new_sentence

train_text=list(map(lambda s: clear_character(s), train_contents))
test_text=list(map(lambda s: clear_character(s), test_contents))
print(train_text[:10])

train_seg_text=list(map(lambda s: jieba.lcut(s), train_text))
test_seg_text=list(map(lambda s: jieba.lcut(s), test_text))

print(train_seg_text[:10])
stop_words_path = "models/stop_words/百度停用词列表.txt"
def get_stop_words():
    file = open(stop_words_path, 'r',encoding='utf-8').read().split('\n')
    return set(file)
stopwords = get_stop_words()
print(stopwords)
# 去掉文本中的停用词
def drop_stopwords(line, stopwords):
    line_clean = []
    for word in line:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean


train_st_text=list(map(lambda s: drop_stopwords(s,stopwords), train_seg_text))
test_st_text=list(map(lambda s: drop_stopwords(s,stopwords), test_seg_text))


le = LabelEncoder()
le.fit(train_labels)


label_train_id=le.transform(train_labels)
label_test_id=le.transform(test_labels)


train_c_text=list(map(lambda s: ' '.join(s), train_st_text))
test_c_text=list(map(lambda s: ' '.join(s), test_st_text))


print(train_c_text[:10])
tfidf_model = TfidfVectorizer(binary=False,token_pattern=r"(?u)\b\w+\b")
train_Data = tfidf_model.fit_transform(train_c_text)
test_Data = tfidf_model.transform(test_c_text)


classifier=LogisticRegression()
classifier.fit(train_Data, label_train_id)
pred = classifier.predict(test_Data)
from sklearn.metrics import classification_report
print(classification_report(label_test_id, pred,digits=4))


joblib.dump(le,'models/le.joblib')
joblib.dump(tfidf_model, 'models/tfidf_model.joblib')
joblib.dump(classifier, 'models/classifier.joblib')

