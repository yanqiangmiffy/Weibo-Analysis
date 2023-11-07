import re
import jieba  # 中文分词包
import jieba.analyse as analyse  # 关键字提取
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud
from weibo_preprocess_toolkit import WeiboPreprocess

from utils import parallelize_on_rows
tqdm.pandas()
preprocess = WeiboPreprocess()


with open('models/stop_words.txt','r',encoding='utf-8') as f:
    stop_words=f.readlines()
    stop_words=[w.strip() for w in stop_words]


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
        text = re.sub(URL_REGEX, "", text)  # 去除网址
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
    def process(self, text):
        text = self.delete_html(text)
        text = self.clean_special(text)
        text=self.delete_num_alpa(text)
        text=self.delte_special_token(text)
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
    text_words = jieba.cut(row['text'], cut_all=False)
    text_words = [w for w in text_words]
    return text_words


if __name__ == '__main__':
    df = pd.read_parquet('data/1_filter.parquet')
    cols = ['wid', 'text', 'forward_nums', 'review_nums', 'upvote', 'domain', 'date']
    df.columns = cols
    # print(df.head(1).values)
    # print(df.columns)
    # print(df.head())
    # print(df.shape)

    text_processor = Processor()
    print(df.isnull().sum())
    df['text_len'] = df['text'].apply(lambda x: len(str(x)))
    print(df['text_len'].describe())

    df['text'] = df['text'].progress_apply(lambda x: text_processor.process(x))
    df['text_len'] = df['text'].apply(lambda x: len(str(x)))
    print(df['text_len'].describe())

    df['text_words'] = parallelize_on_rows(df, cut_df_words, num_of_processes=24)

    # words = cut_df_words(df)
    words = []
    for text_words in df['text_words']:
        words.extend(text_words)

    print(words[:10])

    tfs = jieba_count_tf(words)
    # 形成表格数据
    df = pd.DataFrame(data=list(zip(list(tfs.keys()), list(tfs.values()))), columns=['单词', '频数'])
    df.to_csv('output/word_cnt.csv', index=False)
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
