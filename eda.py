#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: quincy qiang
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2023/11/12 2:42
"""
import pandas as pd
from tqdm import  tqdm
import matplotlib.pyplot as plt
df=pd.read_csv('models2/result_demo_zhongmei2022.csv')
print(df.columns)

def describex(data):
    stats = data.describe()
    skewness = data.skew()
    kurtosis = data.kurtosis()
    skewness_df = pd.DataFrame({'skewness':skewness}).T
    kurtosis_df = pd.DataFrame({'kurtosis':kurtosis}).T
    return stats._append([kurtosis_df,skewness_df])


category_cnt_df=df['category'].value_counts().reset_index()
category_cnt_df.columns = ['category', 'counts']
category_cnt_df.to_csv(f'models2/eda/category数量统计.csv', index=False)

for index,group in df.groupby(by='topic'):
    group.to_csv(f'models2/eda/{index}.csv',index=False)
    words=[]
    for words_str in group['text_words']:
        text_words=eval(words_str)
        words.extend(text_words)
    new_words=[]
    for word in tqdm(words):
        word=word.replace('\\','').replace('n','').replace(' ','').strip()
        if word and len(word)>1:
            new_words.append(word)
    word_cnt=pd.DataFrame({'word':new_words})
    word_cnt_df=word_cnt['word'].value_counts().reset_index()
    word_cnt_df.columns = ['word', 'counts']
    word_cnt_df.to_csv(f'models2/eda/{index}_词频.csv',index=False)
    sentiment_stats = df[['topic', 'sentiment']].groupby('topic').describe().reset_index()
    sentiment_stats.to_excel('models2/eda/情感统计.xlsx')