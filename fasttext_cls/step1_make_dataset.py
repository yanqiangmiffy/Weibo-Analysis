import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


train1 = pd.read_excel('manual_coding.xlsx',sheet_name='weibo_trade_coding')
train2 = pd.read_excel('manual_coding.xlsx',sheet_name='pelosi_taiwan_coding')
train = pd.concat([train1[['content','category']],train2[['content','category']]],axis=0).reset_index(drop=True)
print(train["category"].value_counts())


train["category"]=train["category"].apply(lambda x:5 if x>=5 else x)

train5=train[train['category']==5].sample(n=1000,random_state=42)
train14=train[train['category']!=5]
train = pd.concat([train5[['content','category']],train14[['content','category']]],axis=0).reset_index(drop=True)



lb = LabelEncoder()
train['category'] = lb.fit_transform(train['category'].values)
print(train["category"].value_counts())


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (train_index, valid_index) in enumerate(skf.split(train, train["category"])):
    train.loc[valid_index, "fold"] = fold

for fold in range(5):
    fold_df = train[train['fold'] == fold]
    print(fold_df['category'].value_counts())
    with open(f'fasttext.train_fold{fold}', 'w', encoding='utf-8') as f:
        for idx, row in tqdm(fold_df.iterrows(), total=len(fold_df)):
            f.write(
                f'__label__{row["category"]} {row["content"]}\n')

