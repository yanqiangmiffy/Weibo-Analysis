import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

df = pd.read_parquet('output/task1/weibo_analysis.parquet')


def plot_text_lens(df_sample,mode='Characters'):
    # 创建示例 DataFrame
    df_sample['date'] = pd.to_datetime(df_sample['timestamp'], unit='s')

    # 转换日期类型并提取年份和月份
    df_sample['date'] = pd.to_datetime(df_sample['date'])
    df_sample['year'] = df_sample['date'].dt.year
    df_sample['month'] = df_sample['date'].dt.month

    # 计算每条微博的字数和词数



    # 准备绘图
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 12))

    # 绘制年度平均文本长度（字数）趋势图
    if mode=='Characters':
        df_sample['text_length'] = df_sample['text'].fillna('').progress_apply(len)  # 字数
        yearly_avg_length = df_sample.groupby('year')['text_length'].mean()
        monthly_avg_length = df_sample.groupby(['year', 'month'])['text_length'].mean()

        axes[0].plot(yearly_avg_length.index, yearly_avg_length.values, marker='o', linestyle='-')
        axes[0].set_title(f'Yearly Average Text Length ({mode})')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel(f'Average Length ({mode})')

        # 绘制月度平均文本长度（字数）趋势图
        monthly_avg_length.unstack().plot(ax=axes[1], marker='o', linestyle='-')
        axes[1].set_title(f'Monthly Average Text Length ({mode})')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel(f'Average Length ({mode})')
        axes[1].legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()
    else:
        df_sample['word_count'] = df_sample['text_words'].progress_apply(lambda x: len(eval(x)))  # 词数
        # 计算每年的平均文本长度（字数和词数）
        yearly_avg_word_count = df_sample.groupby('year')['word_count'].mean()

        # 计算每月的平均文本长度（字数和词数）
        monthly_avg_word_count = df_sample.groupby(['year', 'month'])['word_count'].mean()

        axes[0].plot(yearly_avg_word_count.index, yearly_avg_word_count.values, marker='o', linestyle='-')
        axes[0].set_title(f'Yearly Average Text Length ({mode})')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel(f'Average Length ({mode})')

        # 绘制月度平均文本长度（字数）趋势图
        monthly_avg_word_count.unstack().plot(ax=axes[1], marker='o', linestyle='-')
        axes[1].set_title(f'Monthly Average Text Length ({mode})')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel(f'Average Length ({mode})')
        axes[1].legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


def plot_text_forwards(df_sample):
    # 创建示例 DataFrame
    df_sample['date'] = pd.to_datetime(df_sample['timestamp'], unit='s')

    # 转换日期类型并提取年份和月份
    df_sample['date'] = pd.to_datetime(df_sample['date'])
    df_sample['year'] = df_sample['date'].dt.year
    df_sample['month'] = df_sample['date'].dt.month

    # 判断微博是否为原创（forward为null则为原创，即值为1）
    df_sample['original'] = df_sample['forward_uid'].isnull().astype(int)

    # 计算每年原创微博的占比
    yearly_original_ratio = df_sample.groupby('year')['original'].mean()

    # 计算每月原创微博的占比
    monthly_original_ratio = df_sample.groupby(['year', 'month'])['original'].mean().groupby('month').mean()

    # 准备绘图
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    # 绘制年度原创微博占比趋势图
    axes[0].plot(yearly_original_ratio.index, yearly_original_ratio.values, marker='o', linestyle='-')
    axes[0].set_title('Yearly Original Weibo Ratio')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Original Ratio')

    # 绘制月度原创微博占比趋势图
    axes[1].plot(monthly_original_ratio.index, monthly_original_ratio.values, marker='o', linestyle='-')
    axes[1].set_title('Monthly Average Original Weibo Ratio')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Original Ratio')

    plt.tight_layout()
    plt.show()


# plot_text_lens(df,mode="Characters")
# plot_text_lens(df,mode="Words")

plot_text_forwards(df)