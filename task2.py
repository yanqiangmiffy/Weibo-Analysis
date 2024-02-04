import pandas as pd
import os
import glob
from  tqdm import tqdm
result_dir='result'
text_dir='text'

# for csv_file in glob.glob('result/**/*.csv',recursive=True):
#     print(csv_file)

for year in os.listdir(result_dir):
    for csv_file in tqdm(os.listdir(os.path.join(result_dir,year))):
        yearmonth=csv_file[:6]
        csv_file=os.path.join(result_dir,year,csv_file)
        # print(csv_file,yearmonth)
        sentiment_df=pd.read_csv(csv_file)
        sentiment_df = sentiment_df.drop(sentiment_df.index[0]).reset_index(drop=True)
        text_df=pd.read_csv(f'C:/Users/sibo/Desktop/rough_cls_v2/{year}/{yearmonth}_filtered.csv')
        # print(sentiment_df.shape)
        # print(text_df.shape)
        #
        # print(sentiment_df.columns.tolist())
        # print(text_df.columns)

        text_df=text_df.drop(columns=['content', 'content_full', 'user_link'],axis=1)
        df=pd.concat([sentiment_df,text_df],axis=1)
        os.makedirs(f'merge/{year}', exist_ok=True)
        merge_path=f'merge/{year}'
        # print(f'{merge_path}/{yearmonth}_filter_sna.csv')
        df.to_csv(f'{merge_path}/{yearmonth}_filter_sna.csv',index=False)