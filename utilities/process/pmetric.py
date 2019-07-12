from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
import numpy as np
import pandas as pd


def mean_absolute_error(y_true, y_pred,
                        sample_weight=None,
                        multioutput='uniform_average', **kwargs):
    return sk_mean_absolute_error(y_true, y_pred, sample_weight, multioutput)


def group_mean_log_mae(y_true, y_pred, groups, floor=1e-9, **kwargs):
    if type(y_true) == np.ndarray:
        y_true = pd.Series(y_true)

    if type(y_pred) == np.ndarray:
        y_pred = pd.Series(y_pred)

    if type(groups) == np.ndarray:
        groups = pd.Series(groups)

    maes = (y_true - y_pred).abs().groupby(groups).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()