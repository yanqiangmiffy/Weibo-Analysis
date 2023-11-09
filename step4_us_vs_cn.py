import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
pd.set_option('display.max_columns', 20)
demo=pd.read_csv('output/demo_zhongmei2022.csv')
print(demo.columns)
print(demo.head())
print(demo.shape)
print(demo['token_nums'].describe(percentiles=[0.25,0.75]))
print(demo['num_words'].describe(percentiles=[0.25,0.75]))
demo['token_nums'].plot(kind='kde')
demo['num_words'].plot(kind='kde')
plt.legend()
plt.show()


def judge_country(x):
    x=str(x)
    if '美国' in x and '中国' in x:
        return '中国'
    elif '美国' in x:
        return '美国'
    elif '中国' in x:
        return '美国_中国'
    else:
        return '其他'

demo['country']=demo['raw_text'].apply(lambda x:judge_country(x))
demo['country'].value_counts().plot(kind='pie')
print(demo['country'].value_counts())

plt.legend()
plt.show()

demo['polarity']=demo['sentiment'].apply(lambda x:1 if x>0.9 else 0)

us_demo=demo[demo['country']=='美国']
cn_demo=demo[demo['country']=='中国']

print(us_demo['sentiment'].describe())
print(us_demo['polarity'].value_counts())
print(cn_demo['polarity'].value_counts())

us_demo['us_sentiment']=us_demo['sentiment']
cn_demo['cn_sentiment']=cn_demo['sentiment']
us_demo['us_sentiment'].plot(kind='kde')
cn_demo['cn_sentiment'].plot(kind='kde')
plt.legend()
plt.show()


print(demo['topic'].value_counts())

demo['topic'].value_counts().plot(kind='barh')

plt.legend()
plt.show()


print(demo[demo['topic']=='家居']['text'])

demo.hist('sentiment', by='topic')
plt.legend()
plt.show()