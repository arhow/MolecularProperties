import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from utilities.process.pqueue import *
from utilities.process.pnode import *
from utilities.process.putilities import *


def prepare_data(file_folder =  '../data/feature', csv_file_folder = '../data/input', samping_fraction=.001):

    df_train = pd.read_csv(f"{csv_file_folder}/train.csv")
    df_test = pd.read_csv(f"{csv_file_folder}/test.csv")

    df_train = df_train.sample(int(df_train.shape[0]*samping_fraction))
    df_test = df_test.sample(int(df_test.shape[0] * samping_fraction))

    for f in os.listdir(file_folder):
        if (f.endswith('.pkl')) and (not f.startswith('.')):
            if f[:-4].endswith('train'):
                df_feature_i = pd.read_pickle(f'{file_folder}/{f}')
                columns_i = df_feature_i.columns.tolist()
                new_columns = set(columns_i) - set(df_train.columns.tolist())
                df_train = pd.merge(df_train, df_feature_i[list(new_columns) + ['id']], on='id')
                print('train add', f, df_feature_i.shape)
            if f[:-4].endswith('test'):
                df_feature_i = pd.read_pickle(f'{file_folder}/{f}')
                columns_i = df_feature_i.columns.tolist()
                new_columns = set(columns_i) - set(df_test.columns.tolist())
                df_test = pd.merge(df_test, df_feature_i[list(new_columns) + ['id']], on='id')
                print('test add', f, df_feature_i.shape)

    numerics = ['int16', 'int8', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df_train.columns:
        col_type = df_train[col].dtypes
        if not col_type in numerics:
            print(col, df_train[col].unique())
            le = LabelEncoder()
            le.fit(list(df_train[col].values) + list(df_test[col].values))
            df_train[col] = le.transform(list(df_train[col].values))
            df_test[col] = le.transform(list(df_test[col].values))
            print(le.classes_)

    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.fillna(0)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.fillna(0)

    df_train = df_train.rename(columns={'id': 'index', 'scalar_coupling_constant': 'y'})
    df_test = df_test.rename(columns={'id': 'index'})

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_test

if __name__ == "__main__":
    df_train, df_test = prepare_data()
    print(df_train.shape, df_test.shape)

    param = {
        'columns': df_train.columns.drop(['index', 'y']).tolist(),
        'cv': {
            'cls': 'KFold',
            'init': {'n_splits': 5, 'shuffle': True, 'random_state': 42}
        },
        'scaler': {
            'cls': 'StandardScaler', 'init': {}, 'fit': {}
        },
        'model': {
            'cls': 'lgb.LGBMRegressor',
            'init': {
                'learning_rate': 0.35395923077843333,
                'feature_fraction': 0.8840483697334669,
                'bagging_fraction': 0.7017457378676857,
                'min_data_in_leaf': 616,
                'lambda_l1': 0.00013058988949929333,
                'lambda_l2': 0.004991992636437704,
                'max_bin': 74,
                'num_leaves': 255,
                'random_state': 2928,
                'n_jobs': 16
            },
            'fit': {}
        },
        'metric': 'mean_absolute_error'
    }

    mytrial = []
    process_queue = PQueue(df_train, df_test, param, mytrial)
    sort_features = SortFeatureSelectTopNProcess(**{'top_n':200})
    select_topn = RFESelectTopNProcess(**{'n_features_remain':20, 'n_features_to_remove':10})
    remove_useless = RFERemoveUselessFeaturesProcess(**{})
    process_queue.insert_node(sort_features)
    process_queue.insert_node(select_topn)
    process_queue.insert_node(remove_useless)
    result = process_queue.run()
    print(len(process_queue.trial))
    print(process_queue.param)
