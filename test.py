# import pandas as pd
import polars as pl
from pathlib import Path

from ultra import preprocess
from sklearn.model_selection import StratifiedKFold

# from sphinx import quickstart


class CFG:
    # NOW = datetime.datetime.now()
    # os.makedirs(f'./outputs/{NOW}')
    # OUT_DIR = Path(f'./outputs/{NOW}')

    DATA_DIR = Path('./test_datasets/titanic/')

    N_SPLITS    = 5
    RANDOM_SEED = 72

    lgb_params = {
        'objective': 'rmse', # 目的関数. これの意味で最小となるようなパラメータを探します. 
        'learning_rate': 0.1, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
        'reg_lambda': 1., # L2 Reguralization 
        'reg_alpha': .1, # こちらは L1
        'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります
        'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
        'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
        # bagging の頻度と割合
        'subsample_freq': 3,
        'subsample': .9,
        'importance_type': 'gain', # 特徴重要度計算のロジック(後述)
        'random_state': RANDOM_SEED,
    }

train = pl.read_csv(CFG.DATA_DIR / 'train.csv')
test = pl.read_csv(CFG.DATA_DIR / 'test.csv')


target_cols = train.select([pl.col(pl.Utf8), pl.col(pl.Categorical), pl.col(pl.Object)]).columns
print(target_cols)

train = preprocess.count_encoding(train, target_cols)
train = preprocess.onehot_encoding(train, target_cols)
train = preprocess.label_encoding(train, target_cols)

y = 'Survived'
folds = StratifiedKFold(n_splits=CFG.N_SPLITS, random_state=CFG.RANDOM_SEED, shuffle=True)
train, test = preprocess.target_encoding(train, test, folds, target_cols, y)




# print(train)




# # train = pd.read_csv(DATA_DIR / 'train.csv')
# print(train)
# obj_col = train.select_dtypes(include=object).columns.tolist()
# print(obj_col)

# train = preprocess.count_encoding(train, obj_col)
# print(train)

# train = utable.Utable(train)

# train.count_encoding(obj_col)