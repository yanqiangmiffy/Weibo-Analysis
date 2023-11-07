import pandas as pd
import os
from tqdm import  tqdm
data_dir="E:\\Jiajun\\filtered"

for csv_file in tqdm(os.listdir(data_dir)):
    data=[]
    with open(os.path.join(data_dir,csv_file),'r',encoding='utf-8') as f:
        for line in f.readlines():
            # print(line.strip())
            # print(len((line.strip().split('	'))))
            sample=line.strip().split(',')
            if len(sample)==7:
                data.append(sample)
    # print(data[:2])
    print(len(data))

    df = pd.DataFrame(data[1:])
    cols = ['wid', 'text', 'forward_nums', 'review_nums', 'upvote', 'domain', 'date']
    df.columns=cols
    file_name=csv_file.replace('.csv','')
    df.to_parquet(f'data/{file_name}.parquet',index=False)