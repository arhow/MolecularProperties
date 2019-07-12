import pandas as pd
import numpy as np
import os
import copy
import datetime

import eli5
from eli5.sklearn import PermutationImportance

from utilities.process import putilities as processutil




def sk_process(df_train, param, message, df_test=None, trial=None, is_output_feature_importance=False, trial_level=0):
    """
    >>>param = {
    >>>    'columns': columns,
    >>>    'cv': {
    >>>        'cls': 'KFold',
    >>>        'init': {'n_splits': 5, 'shuffle': True, 'random_state': 42}
    >>>    },
    >>>    'scaler': {
    >>>        'cls': 'StandardScaler', 'init': {}, 'fit': {}
    >>>    },>>>
    >>>    'model': {
    >>>        'cls': 'lgb.LGBMRegressor',
    >>>        'init': {
    >>>            'learning_rate': 0.35395923077843333,
    >>>             'feature_fraction': 0.8840483697334669,
    >>>            'bagging_fraction': 0.7017457378676857,
    >>>            'min_data_in_leaf': 616,
    >>>            'lambda_l1': 0.00013058988949929333,
    >>>            'lambda_l2': 0.004991992636437704,
    >>>            'max_bin': 74,
    >>>            'num_leaves': 64,
    >>>            'random_state': 2928,
    >>>            'n_jobs': 16
    >>>        },
    >>>        'fit': {}
    >>>    },
    >>>    'metric': 'mean_absolute_error'
    >>>}
    :param df_train:
    :param param:
    :param message:
    :param df_test:
    :param trial:
    :param is_output_feature_importance:
    :param trial_level:
    :return:
    """

    columns = param['columns']

    assert 'y' in df_train.columns.tolist(), 'y is not in df_train'
    assert 'index' in df_train.columns.tolist(), 'index is not in df_train'
    assert 'index' not in param['columns'], 'index is in features'
    assert 'y' not in param['columns'], 'y is in features'
    assert 'label' not in param['columns'], 'label is in features'
    assert 'group' not in param['columns'], 'group is in features'
    assert (type(trial) == list) | (trial == None), 'trial is neither list nor none'
    assert len(columns) != 0, 'columns size is 0'

    df_test_pred = None
    if type(df_test) == pd.DataFrame:
        assert 'index' in df_test.columns.tolist(), 'index is not in df_test'
        df_test_pred = pd.concat([df_test_pred, df_test[['index']]], axis=1)

    CV = processutil._str2class(param['cv']['cls'])
    MODEL = processutil._str2class(param['model']['cls'])
    if 'scaler' in param:
        SCALER = processutil._str2class(param['scaler']['cls'])
    metric = processutil._str2class(param['metric'])

    history = []
    df_valid_pred = pd.DataFrame()
    df_feature_importances_i_list = []

    # StratifiedKFold, KFold, RepeatedKFold,TimeSeriesSplit, GroupKFold
    if 'splits' in param['cv']:
        splits = param['cv']['splits']
    else:
        cv = CV(**param['cv']['init'])
        if param['cv']['cls'] == 'StratifiedKFold':
            assert 'label' in df_train.columns.tolist(), 'label is not in df_train'
            splits = list(cv.split(df_train, df_train['label']))
        elif param['cv']['cls'] == 'GroupKFold':
            assert 'group' in df_train.columns.tolist(), 'group is not in df_train'
            splits = list(cv.split(df_train, groups=df_train['group']))
        else:
            splits = list(cv.split(df_train))

    for fold_n, (train_index, valid_index) in enumerate(splits):

        X_train, X_valid = df_train[columns].values[train_index, :], df_train[columns].values[valid_index, :]
        y_train, y_valid = df_train['y'].values[train_index], df_train['y'].values[valid_index]

        if 'scaler' in param:
            scaler = SCALER(**param['scaler']['init'])
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

        model = MODEL(**param['model']['init'])
        model.fit(X_train, y_train, **param['model']['fit'])

        y_valid_pred = model.predict(X_valid)
        y_train_pred = model.predict(X_train)

        original_index = df_train['index'].values[valid_index]
        df_valid_pred_i = pd.DataFrame \
            ({'index': original_index, 'predict': y_valid_pred, 'fold_n': np.zeros(y_valid_pred.shape[0]) + fold_n})
        df_valid_pred = pd.concat([df_valid_pred, df_valid_pred_i], axis=0)

        if is_output_feature_importance:
            df_feature_importances_i = pd.DataFrame({'feature': columns, 'model_weight': model.feature_importances_})
            df_feature_importances_i = df_feature_importances_i.sort_values(by=['feature'])
            df_feature_importances_i = df_feature_importances_i.reset_index(drop=True)
            perm = PermutationImportance(model, random_state=42).fit(X_valid, y_valid)
            df_feature_importances_i2 = eli5.explain_weights_dfs(perm, feature_names=columns, top=len(columns))['feature_importances']
            df_feature_importances_i2 = df_feature_importances_i2.sort_values(by=['feature'])
            df_feature_importances_i2 = df_feature_importances_i2.reset_index(drop=True)
            df_feature_importances_i = pd.merge(df_feature_importances_i, df_feature_importances_i2, on='feature')
            df_feature_importances_i_list.append(df_feature_importances_i)

        if type(df_test) == pd.DataFrame:
            X_test = df_test[columns].values
            if 'scaler' in param:
                X_test = scaler.transform(X_test)
            y_test_pred = model.predict(X_test)
            df_test_pred_i = pd.DataFrame({fold_n: y_test_pred})
            df_test_pred = pd.concat([df_test_pred, df_test_pred_i], axis=1)

        history.append \
            ({'fold_n': fold_n, 'train': metric(y_train, y_train_pred, group=df_train['group']), 'valid': metric(y_valid, y_valid_pred, group=df_train['group'])})

    df_his = pd.DataFrame(history)

    df_feature_importances = None
    if is_output_feature_importance:
        df_feature_importances = df_feature_importances_i_list[0]
        for idx, df_feature_importances_i in enumerate(df_feature_importances_i_list[1:]):
            df_feature_importances = pd.merge(df_feature_importances, df_feature_importances_i, on='feature', suffixes=('', idx + 1))

    df_valid_pred = df_valid_pred.sort_values(by=['index'])
    df_valid_pred = df_valid_pred.reset_index(drop=True)

    if type(df_test) == pd.DataFrame:
        df_test_pred = df_test_pred.sort_values(by=['index'])
        df_test_pred = df_test_pred.reset_index(drop=True)

    if type(trial) == list:
        datetime_ = datetime.datetime.now()
        val_metric_mean = np.mean(df_his.valid)
        val_metric_std = np.std(df_his.valid)
        train_metric_mean = np.mean(df_his.train)
        train_metric_std = np.std(df_his.train)

        trial_i_d_ = {'datetime': datetime_, 'message': message, 'val_metric_mean': val_metric_mean,
                      'train_metric_mean': train_metric_mean, 'val_metric_std': val_metric_std, 'train_metric_std': train_metric_std,
                      'trn_val_metric_diff': val_metric_mean - train_metric_mean,
                      'df_feature_importances': df_feature_importances ,'param': param.copy(),
                      'nfeatures': len(columns)}
        if trial_level > 0:
            trial_i_d_ = {'df_his': df_his, 'df_valid_pred': df_valid_pred, 'df_test_pred': df_test_pred, **trial_i_d_}
        trial.append(trial_i_d_)

    return df_his, df_feature_importances, df_valid_pred, df_test_pred

def sort_feature_importances(df_feature_importances, key='average_model_weight'):
    df_feature_importances['average_permutation_weight'] = df_feature_importances[
        [col for col in df_feature_importances.columns.tolist() if ('weight' in col) & ('model' not in col)]].mean(
        axis=1)
    df_feature_importances['average_model_weight'] = df_feature_importances[
        [col for col in df_feature_importances.columns.tolist() if ('model_weight' in col)]].mean(axis=1)
    df_feature_importances = df_feature_importances.sort_values(by=[key], ascending=False)
    sorted_columns = df_feature_importances.feature.tolist()
    return sorted_columns


def ref(df_train, param, trial, message, df_test=None, n_features_remain=10, n_features_to_remove=10, key='average_model_weight'):
    param_i = param.copy()

    while True:
        df_his, df_feature_importances, df_valid_pred, df_test_pred = sk_process(df_train, param_i, df_test=df_test, trial=trial, is_output_feature_importance=True, message=message)
        sorted_columns = sort_feature_importances(df_feature_importances, key)
        if (len(sorted_columns) <= n_features_remain)|(len(sorted_columns)-n_features_to_remove<1):
            break
        else:
            param_i['columns'] = sorted_columns[:-n_features_to_remove]
    return


def width_frist_rfe(df_train, param, trial, score, message, df_test=None):

    param_ = copy.deepcopy(param)
    columns_ = param_['columns']
    best_score = score
    best_param = param_
    for col in columns_:
        param_['columns'] = list(set(columns_) - set([col]))
        df_his, df_feature_importances, df_valid_pred, df_test_pred = sk_process(df_train, param_, df_test=df_test, trial=trial, is_output_feature_importance=False, message=message)
        val_mae_mean = np.mean(df_his.valid)
        if val_mae_mean<best_score:
            best_score = val_mae_mean
            best_param = copy.deepcopy(param_)

    if best_score < score:
        width_frist_rfe(df_train, best_param, trial, best_score, message, df_test)

    return