import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold,TimeSeriesSplit, GroupKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error


def _str2class(s):
    if s in globals() and isinstance(globals()[s], type):
            return globals()[s]
    if isinstance(eval(s), type):
        return eval(s)
    if callable(eval(s)):
        return eval(s)
    return None