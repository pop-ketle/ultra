import numpy as np 
import pandas as pd
import polars as pl

from sklearn import preprocessing


# class Utable:
#     def __init__(self, df):
#         super().__init__()
#         self.df = df
    
def count_encoding(df, target_cols):
    for c in target_cols:
        _df = df[c].value_counts().rename({'counts': f'CE_{c}'})
        df = df.join(_df, on=[c], how='left')
    return df

def onehot_encoding(df, target_cols):
    for c in target_cols:
        _df = df[c].to_dummies()
        _df = _df.with_columns(pl.all().prefix('OH_'))
        df = pl.concat([df, _df], how='horizontal')
    return df

def label_encoding(df, target_cols):
    le = preprocessing.LabelEncoder()
    for c in target_cols:
        df = df.with_columns(pl.Series(le.fit_transform(df[c])).alias(f'LE_{c}'))
    return df  

def target_encoding(train, test, folds, target_cols, y):
    # 学習データ全体でカテゴリにおけるyの平均を計算
    for c in target_cols:
        target_mean = train.groupby(c).mean()[[c,y]].rename({y: f'TE_{c}'})
        test = test.join(target_mean, on=[c], how='left')

        # 変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train.shape[0])
        for train_idx, test_idx in folds.split(train, train[y]):
            target_mean = train[train_idx].groupby(c).mean()[[c,y]].rename({y: f'TE_{c}'})
            # バリデーションデータについて、変換後の値を一時配列に格納
            tmp[test_idx] = train[test_idx].join(target_mean, on=[c], how='left')[f'TE_{c}']
        train = train.with_columns(pl.Series(tmp).alias(f'TE_{c}'))

    return train, test


    




# class FeedbackDataset:
#     def __init__(self, samples, max_len, tokenizer):
#         self.samples = samples
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#         self.length = len(samples)