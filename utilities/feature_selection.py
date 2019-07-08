import types
import pandas as pd
import numpy as np
import os
import datetime
from IPython.lib import kernel

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from fastFM import als, mcmc, sgd
from rgf.sklearn import RGFRegressor
#from pyfm import pylibfm

from scipy import sparse

import eli5
from eli5.sklearn import PermutationImportance

import copy


def _evaluate(df_feature_importances, key='average_model_weight'):
    df_feature_importances['average_permutation_weight'] = df_feature_importances[
        [col for col in df_feature_importances.columns.tolist() if ('weight' in col) & ('model' not in col)]].mean(axis=1)
    df_feature_importances['average_model_weight'] = df_feature_importances[
        [col for col in df_feature_importances.columns.tolist() if ('model_weight' in col)]].mean(axis=1)
    df_feature_importances = df_feature_importances.sort_values(by=[key], ascending=False)
    sorted_columns = df_feature_importances.feature.tolist()
    return sorted_columns

def select_features_(df_train, param, trial, df_test=None, nfeats_best=10, nfeats_removed_per_try=10, key='average_model_weight', remark=None):
    param_i = param.copy()
    while True:
        df_his, df_feature_importances, df_valid_pred, df_test_pred = sk_process(df_train, param_i, df_test=df_test, trial=trial, is_output_feature_importance=True, remark=remark)
        sorted_columns = _evaluate(df_feature_importances, key)
        if (len(sorted_columns) <= nfeats_best)|(len(sorted_columns)-nfeats_removed_per_try<1):
            break
        else:
            param_i['columns'] = sorted_columns[:-nfeats_removed_per_try]
    return

def width_frist_rfe(df_train, param, trial, score, df_test=None, remark=None):

    param_ = copy.deepcopy(param)
    columns_ = param_['columns']
    best_score = score
    best_param = param_
    for col in columns_:
        param_['columns'] = list(set(columns_) - set([col]))
        df_his, df_feature_importances, df_valid_pred, df_test_pred = sk_process(df_train, param_, df_test=df_test, trial=trial, is_output_feature_importance=False, remark=remark)
        val_mae_mean = np.mean(df_his.valid)
        if val_mae_mean<best_score:
            best_score = val_mae_mean
            best_param = copy.deepcopy(param_)

    if best_score < score:
        width_frist_rfe(df_train, best_param, trial, best_score, df_test, remark=remark)

    return

def revert_rfe(df_train, param, sorted_columns, df_test, trial, start_columns, limit=None, remark=None):

    # init cv_score and try only base feature
    selected_columns = copy.deepcopy(start_columns)
    if type(limit) == type(None):
        limit = len(sorted_columns)
    args = copy.deepcopy(param)
    args['columns'] = selected_columns
    df_his,  df_feature_importances, df_valid_pred, df_test_pred =  sk_process(df_train, args, df_test = df_test, trial=trial, remark=remark)
    val_mae_mean = np.mean(df_his.valid)
    cv_score = val_mae_mean

    # add feature one by one and check cv score change
    for idx,col in enumerate(sorted_columns):
#         if idx in start_column_index:
#             continue
        args = copy.deepcopy(param)
        args['columns'] = list(set(selected_columns + [col]))
        df_his,  df_feature_importances, df_valid_pred, df_test_pred =  sk_process(df_train, args, df_test = df_test, trial=trial, remark=remark)
        val_mae_mean = np.mean(df_his.valid)
        if val_mae_mean < cv_score:
            selected_columns.append(col)
            cv_score = val_mae_mean
        if len(selected_columns) >= limit:
            break

    return selected_columns

def blacklist_merge(df, columns=None, base_correlation_coefficient=.9):

    if type(columns)==type(None):
        columns = df.columns.tolist()
    bcc_ = base_correlation_coefficient
    X = df_train[columns].values
    X = StandardScaler().fit_transform(X)
    df_norm = pd.DataFrame(X, columns=columns)
    df_corr = df_norm.corr()

    black_lst = []
    group = {}
    for col in columns:
        if col in black_lst:
            continue
        group[col] = list(df_corr[(df_corr[col]>=bcc_)|(df_corr[col]<=-bcc_)].index)
        black_lst +=  group[col]
    return group

def bubble_merge(df, columns=None, base_correlation_coefficient=.9, coverage_rate=.9):

    def is_similar(group1, group2):
        assert type(group1)==list, 'group1 should be a list'
        assert type(group2)==list, 'group2 should be a list'
        total_units = group1 + group2
        unique_units = list(set(total_units))
        common_parts = [col for col in unique_units if total_units.count(col)==2]
        if (len(common_parts)/len(group1) >= coverage_rate) | (len(common_parts)/len(group2) >= coverage_rate):
            return True
        else:
            return False

    def merge_group(original_group):
        group = original_group.copy()
        merged_group = group
        dict_list_ = list(group.items())
        is_merged = False

        index1 = 1
        for k1, v1 in dict_list_[:-1]:
            for k2,v2 in dict_list_[index1:]:
                    if is_similar(v1, v2):
                        group[k1] = list(set(v1 + v2))
                        del group[k2]
                        merged_group = merge_group(group)
                        is_merged = True
                        break
            if is_merged:
                break
            index1 += 1
        return merged_group

    if type(columns)==type(None):
        columns = df.columns.tolist()
    bcc_ = base_correlation_coefficient
    X = df[columns].values
    X = StandardScaler().fit_transform(X)
    df_norm = pd.DataFrame(X, columns=columns)
    df_corr = df_norm.corr()

    group = {}
    for col in columns:
        group[col] = list(df_corr[(df_corr[col]>=bcc_)|(df_corr[col]<=-bcc_)].index)

    return merge_group(group)