import jieba  # 中文分词包
import jieba.analyse as analyse  # 关键字提取
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
from tqdm import  tqdm
from utils import parallelize_on_rows


# 统计词频函数
def jieba_count_tf(words, is_filter_stop_words=True):
    stop_words = []
    if is_filter_stop_words:
        # 获取Jieba库载入的停用词库
        stop_words = analyse.default_tfidf.stop_words
    freq = {}
    for w in tqdm(words):
        if len(w.strip()) < 2 or w.lower() in stop_words:
            continue
        freq[w] = freq.get(w, 0.0) + 1.0
    # 总词汇数量
    total = sum(freq.values())
    return freq


def cut_df_words(row):
    text_words = jieba.cut(row['text'], cut_all=False)
    text_words=[w for w in text_words]
    return text_words

if __name__ == '__main__':
    df = pd.read_parquet('data/1_filter.parquet')
    cols = ['wid', 'text', 'forward_nums', 'review_nums', 'upvote', 'domain', 'date']
    df.columns = cols
    # print(df.head(1).values)
    # print(df.columns)
    # print(df.head())
    # print(df.shape)

    print(df.isnull().sum())
    df['text_len'] = df['text'].apply(lambda x: len(str(x)))
    print(df['text_len'].describe())

    df['text_words'] = parallelize_on_rows(df, cut_df_words,num_of_processes=24)

    # words = cut_df_words(df)
    words=[]
    for text_words in df['text_words']:
        words.extend(text_words)

    print(words[:10])

    tfs = jieba_count_tf(words, is_filter_stop_words=False)
    # 形成表格数据
    df = pd.DataFrame(data=list(zip(list(tfs.keys()), list(tfs.values()))), columns=['单词', '频数'])
    df.to_csv('word_cnt.csv',index=False)
    # # 关键字分析
    # tfidf_fre = analyse.extract_tags(text, topK=100, withWeight=True, allowPOS=(), withFlag=True)
    # # 形成表格数据
    # df = pd.DataFrame(data=tfidf_fre, columns=['单词', 'tf-idf'])

    # 生成词云图展示
    # alice_mask = np.array(Image.open('test.png'))  # 使用图片作为背景
    font_path = 'C:\Windows\Fonts\simfang.ttf'  # 设置文本路径
    # 创建图云对象
    wc = WordCloud(font_path=font_path, background_color='white',
                   # mask=alice_mask,
                   max_words=400,
                   width=800, height=400, stopwords=None)
    wc.fit_words(dict(zip(df['单词'], df['频数'])))  # 输入词频字典，或者 {词：tf-idf}
    wc_img = wc.to_image()  # 输出为图片对象
    wc.to_file("alice.png")
