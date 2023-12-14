import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
from weibo_analysis import WeiboAnalysis

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def concat():
    base_dir = r'C:\Users\sibo\Desktop\processed'

    dfs = []

    for dir in os.listdir(base_dir):
        if dir.startswith('2'):
            for csv_file in os.listdir(os.path.join(base_dir, dir)):
                if 'png' not in csv_file and '_cnt' not in csv_file:
                    print(os.path.join(base_dir, dir, csv_file))
                    df = pd.read_csv(os.path.join(base_dir, dir, csv_file))
                    dfs.append(df)
    print(len(dfs))
    df = pd.concat(dfs, axis=0).reset_index(drop=True)

    print(df.shape)

    df.to_parquet('output/weibo_analysis.parquet', index=None)


def process():
    df = pd.read_parquet('output/weibo_analysis.parquet')
    print(df)
    df['time'] = df['time'].apply(
        lambda x: x.strip().replace(r'\n', '').replace(r'\n                        ', '').replace(r'\n', '').strip())
    df[['year', 'month', 'timestamp']] = df.apply(lambda row: get_year_month(row['time']), axis=1, result_type='expand')
    df.to_parquet('output/task1/weibo_analysis.parquet', index=None)

    # ['_id', 'nickname', 'time', 'from', 'content', 'content_full',
    #        'user_link', 'forward_nickname', 'forward_uid', 'forward_content',
    #        'text', 'raw_text', 'token_nums', 'text_words', 'sentiment', 'topic',
    #        'num_words', 'category']

    for topic in df['topic'].unique():
        print(topic)
        df[df['topic'] == topic].to_parquet(f'output/task1/{topic}.parquet', index=None)


def get_year_month(date_str):
    # 字符串
    # date_str = "2022年12月21日 00:38"

    # 转换为datetime对象
    date_obj = datetime.strptime(date_str, "%Y年%m月%d日 %H:%M")

    # 提取年份和月份
    year = date_obj.year
    month = date_obj.month

    # 转换为时间戳
    timestamp = datetime.timestamp(date_obj)

    return year, month, timestamp


def plot_weibo_month(df, is_vs=False,save_img=''):
    # 示例数据（在实际应用中，您需要替换为您的实际数据）
    # data = {
    #     'date': ['2022-01-01', '2022-01-02', '2022-02-01', '2022-02-02', '2022-03-01'],
    #     'num_posts': [10, 20, 15, 25, 30]
    # }
    # df = pd.DataFrame(data)

    # 将'date'列转换为datetime对象，并提取年月
    # df['date'] = pd.to_datetime(df['timestamp'])
    # df['month'] = df['date'].dt.to_period('M')
    # print(df['date'])
    # print(df['month'])
    # 按月份汇总微博条数
    monthly_counts = df.groupby('year_month')['_id'].count()
    if is_vs:
        monthly_counts = df.groupby('year_month')['_id'].count() / df[df['year_month'] == '2011-8'].shape[0]
    # 绘制趋势图
    plt.figure(figsize=(10, 6))
    monthly_counts.plot(kind='line', marker='o')
    plt.title('微博条数月度趋势图')
    plt.xlabel('月份')
    plt.ylabel('微博条数')
    plt.grid(True)
    plt.savefig(save_img,format='png',dpi=300)
    # plt.show()


def plot_weibo_week(df, is_vs,save_img=''):
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')

    week_dfs = []
    for year in df['year'].unique():
        print(year)
        # 为每条微博创建一个星期标识
        week_weibo_df = df[df['year'] == year]
        week_weibo_df['week'] = week_weibo_df['date'].dt.isocalendar().week

        # 统计每个星期的微博个数
        weekly_counts = week_weibo_df.groupby('week').size().reset_index(name='num_posts')
        weekly_counts['week'] = str(int(year)) + '-week-' + weekly_counts['week'].astype(str)
        week_dfs.append(weekly_counts)

    week_df = pd.concat(week_dfs, axis=0).reset_index(drop=True)
    # print(week_df)
    if is_vs:
        cnt = week_df[week_df['week'] == '2011-week-31']['num_posts'].values[0]
        print(cnt)
        week_df['num_posts'] = week_df['num_posts'] / cnt
    # print(week_df['num_posts'])
    week_df['week'] = week_df.index
    # 绘制趋势图
    plt.figure(figsize=(20, 10))
    plt.plot(week_df['week'], week_df['num_posts'], marker='o', color='teal', linewidth=2, markersize=8)
    # plt.xticks([])

    # plt.xticks(rotation=80)
    # 设置图表标题和坐标轴标签
    plt.title('每周微博条数趋势', fontsize=2)
    plt.xlabel('星期', fontsize=2)
    plt.ylabel('微博条数', fontsize=2)
    # 添加次要刻度线
    # plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))  # 在每个主要刻度之间添加两个次要刻度
    # plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))  # 同样操作应用于y轴

    # 显示网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(save_img,format='png',dpi=300)

    # 显示图形
    # plt.show()


def plot_weibo_day(df, is_vs=False,save_img=''):
    from matplotlib.ticker import AutoMinorLocator

    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year_month_day'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
    print(df['year_month_day'])
    monthly_counts = df.groupby(['year_month_day'])['_id'].count()
    if is_vs:
        cnt = df[df['year_month_day'] == '2011-8-3'].size
        print(cnt)
        monthly_counts = monthly_counts / cnt
    # 绘制趋势图
    plt.figure(figsize=(25, 15))
    monthly_counts.plot(kind='line', marker='o')
    plt.title('微博条数每天趋势图')
    plt.xlabel('日期')
    plt.ylabel('微博条数')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))  # 在每个主要刻度之间添加两个次要刻度
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))  # 同样操作应用于y轴
    plt.grid(True)
    plt.savefig(save_img,format='png',dpi=300)

    # plt.show()


def plot_text_lens(df_sample, mode='Characters'):
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
    if mode == 'Characters':
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

df = pd.read_parquet('output/task1/weibo_analysis.parquet')
df['year_month'] = df['year'].astype(int).astype(str) + '-' + df['month'].astype(int).astype(str)


for topic in df['topic'].unique():
    topic_df=df[df['topic']==topic].reset_index(drop=True)
    plot_weibo_month(topic_df,is_vs=False,save_img=f'output/task1/2-a)-{topic}.png')
    plot_weibo_month(topic_df,is_vs=True,save_img=f'output/task1/2-b)-{topic}.png')

    plot_weibo_week(topic_df, is_vs=False, save_img=f'output/task1/2-a)-{topic}.png')
    plot_weibo_week(topic_df, is_vs=True, save_img=f'output/task1/2-b)-{topic}.png')

    plot_weibo_day(topic_df, is_vs=False, save_img=f'output/task1/2-a)-{topic}.png')
    plot_weibo_day(topic_df, is_vs=True, save_img=f'output/task1/2-b)-{topic}.png')
    time.sleep(1)
# wa = WeiboAnalysis()
# wa.parse_words(df[df['topic'] == '时政'], save_cnt='output/task1/4_word_cnt.csv',
#                save_img='output/task1/4_word_cnt.png')
