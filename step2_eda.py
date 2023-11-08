import os
import re
import jieba  # 中文分词包
import jieba.analyse as analyse  # 关键字提取
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud
from weibo_preprocess_toolkit import WeiboPreprocess
from snownlp import SnowNLP

from utils import parallelize_on_rows
from jieba import analyse
from utils import TopicClassifier

tqdm.pandas()
preprocess = WeiboPreprocess()
tc=TopicClassifier()

stop_words=[]
for words_file in os.listdir('models/stop_words'):
    with open(f'models/stop_words/{words_file}', 'r', encoding='utf-8') as f:
        tmp=f.readlines()
        stop_words+=[w.strip() for w in tmp]

stop_words=list(set(stop_words))
class Processor():
    def clean_special(self, text):
        """
        清晰
        网页
        @用户名（包括转发路径上的其他用户名）
        表情符号(用[]包围)
        话题(用#包围)
        :param text:
        :return:
        """
        text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
        text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
        # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
        URL_REGEX = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s('
            r')<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            re.IGNORECASE)
        # text = re.sub(URL_REGEX, "", text)  # 去除网址 会出现死循环
        text = text.replace("转发微博", "")  # 去除无意义的词语
        text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
        return text.strip()

    # 过滤文本中的html链接等
    def delete_html(self, text_context):
        re_tag = re.compile('</?\w+[^>]*>')  # HTML标签
        new_text = re.sub(re_tag, '', text_context)
        new_text = re.sub(",+", ",", new_text)  # 合并逗号
        new_text = re.sub(" +", " ", new_text)  # 合并空格
        new_text = re.sub("[...|…|。。。]+", "...", new_text)  # 合并句号
        new_text = re.sub("-+", "--", new_text)  # 合并-
        text_content = re.sub("———+", "———", new_text)  # 合并-
        return text_content
    def delete_num_alpa(self,text):
        text=re.sub(r'[0-9]+', '', text)
        return  text

    def delte_special_token(self,text):
        """
        去除标点符号以及href连接
        :param text:
        :return:
        """
        results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', re.S)
        text = re.sub(results, '', text)
        remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
        text = re.sub(remove_chars, "", text)
        return text
    def replace_special(self,text):
        text=text.replace('\u200b','')
        return text
    def process(self, text):
        # print("----"*10)
        # print(text)
        # text=text.rstrip('http')
        text = self.delete_html(text)
        # print("delete_html",text)
        text = self.clean_special(text)
        # print("clean_special",text)

        text=self.delete_num_alpa(text)
        # print("delete_num_alpa",text)

        text=self.delte_special_token(text)
        # print("delte_special_token",text)

        text=self.replace_special(text)
        # print("replace_special",text)

        return text


# 统计词频函数
def jieba_count_tf(words):
    # stop_words = []
    # if is_filter_stop_words:
    #     # 获取Jieba库载入的停用词库
    #     stop_words = analyse.default_tfidf.stop_words
    freq = {}
    for w in tqdm(words):
        if len(w.strip()) < 2 or w.lower() in stop_words:
            continue
        freq[w] = freq.get(w, 0.0) + 1.0
    # 总词汇数量
    total = sum(freq.values())
    return freq


def cut_df_words(row):
    # text_words = jieba.cut(row['text'], cut_all=False)
    text_words = jieba.lcut_for_search(row['text'])
    # text_words = [w for w in text_words]
    # text_words=[w for w in analyse.tfidf(row['text'])]
    return text_words

def infer_sentiment(row):
    sn=SnowNLP(row['text'])
    return sn.sentiments

def infer_topic(row):
    sn=tc.predict(" ".join(row['text_words']))
    return sn
# def infer_topic(text_words):
#     sn=tc.predict(" ".join(text_words))
#     return sn
if __name__ == '__main__':
    df = pd.read_parquet('data/10_filter.parquet')
    print(df.columns)
    print(df.head())
    print(df.shape)

    text_processor = Processor()
    print(df.isnull().sum())
    df['raw_text'] = df['text'].copy()
    df['token_nums'] = df['text'].apply(lambda x: len(str(x)))
    df=df.sort_values(by=["token_nums"],ascending=False)
    df['text'] = df['text'].progress_apply(lambda x: text_processor.process(x))

    df['text_words'] = parallelize_on_rows(df, cut_df_words, num_of_processes=24)
    print("jieba cut done!")
    df['sentiment'] = parallelize_on_rows(df, infer_sentiment, num_of_processes=24)
    print("sentiment done!")
    df['topic'] = parallelize_on_rows(df, infer_topic, num_of_processes=12)
    print("topic done!")
    # df['topic'] = df['text_words'].progress_apply(lambda x: infer_topic(x))

    df['num_words'] = df['text_words'].apply(lambda x: len(x))

    df.to_csv('output/demo.csv',index=False)
    words = []
    for text_words in df['text_words']:
        words.extend(text_words)

    print(words[:10])

    tfs = jieba_count_tf(words)
    # 形成表格数据
    df = pd.DataFrame(data=list(zip(list(tfs.keys()), list(tfs.values()))), columns=['单词', '频数'])
    df=df.sort_values(by='频数',ascending=False)
    df.to_csv('output/word_cnt.csv', index=False)




    font_path = 'C:\Windows\Fonts\simfang.ttf'  # 设置文本路径
    # 创建图云对象
    wc = WordCloud(font_path=font_path, background_color='white',
                   # mask=alice_mask,
                   max_words=800,
                   width=1600, height=800, stopwords=None)
    wc.fit_words(dict(zip(df['单词'], df['频数'])))  # 输入词频字典，或者 {词：tf-idf}
    wc_img = wc.to_image()  # 输出为图片对象
    wc.to_file("output/alice.png")
