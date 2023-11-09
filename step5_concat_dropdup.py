import pandas as pd

csv_files = [
    'E:/Jiajun/version231108/raw/meiguo2020.csv',
    'E:/Jiajun/version231108/raw/meiguo2021.csv',
    'E:/Jiajun/version231108/raw/meiguo2022.csv',
    'E:/Jiajun/version231108/raw/zhongmei2020.csv',
    'E:/Jiajun/version231108/raw/zhongmei2021.csv',
    'E:/Jiajun/version231108/raw/zhongmei2022.csv',
]

for csv_file in csv_files:
    print(csv_file)
    df = pd.read_csv(csv_file)

    # print(df.columns)
    # print(df.isnull().sum())
    print(df.shape)
    df['content_full'] = df['content_full'].fillna('')

    df['content_concat'] = df['content'] + ' ' + df['content_full']

    df = df.drop_duplicates(subset=['time', 'nickname']).reset_index(drop=True)
    print(df.shape)
    csv_name=csv_file.split('/')[-1]
    df.to_csv(f'E:/Jiajun/version231108/processed/{csv_name}',index=None)