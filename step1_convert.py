import pandas as pd
import os
from tqdm import tqdm
import gc
def convert2parquet():
    data_dir="E:\\Jiajun\\filtered"

    for csv_file in tqdm(os.listdir(data_dir)):
        print(csv_file)
        df=pd.read_csv(os.path.join(data_dir,csv_file))
        # print(df)
        print(df.shape)
        file_name=csv_file.replace('.csv','')
        df.to_parquet(f'data/{file_name}.parquet',index=False)

def concat_parquet():
    dfs=[]
    for par_file in tqdm(os.listdir('data')[:3]):
        try:
            tmp=pd.read_parquet(f'data/{par_file}')
            dfs.append(tmp)
            del tmp
            gc.collect()
        except Exception as e:
            print(par_file,e)
    df=pd.concat(dfs,axis=0).reset_index(drop=True)

    df.to_parquet('data/df.parquet',index=False)


if __name__ == '__main__':
    concat_parquet()