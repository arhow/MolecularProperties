from utilities.process.process import *

class PNode(object):

    def __init__(self, next=None, previous=None):
        self.next = None
        self.previous = None
        return

    def run(self, *args, **kwargs):
        return {}


class SortFeatureSelectTopNProcess(PNode):

    def __init__(self, top_n, message=None, next=None, previous=None):

        self.top_n = top_n
        self.message = message
        if type(self.message) == type(None):
            self.message = f'{self.__class__.__name__} get top{top_n} features'

        super(SortFeatureSelectTopNProcess, self).__init__()
        return

    def run(self, df_train, df_test, param, trial):
        df_his, df_feature_importances, df_valid_pred, df_test_pred = sk_process(df_train, param.copy(), self.message, df_test=None, trial=trial, is_output_feature_importance=True, trial_level=0)
        sorted_columns = sort_feature_importances(df_feature_importances, 'average_permutation_weight')
        sorted_columns = sorted_columns[:200] if len(sorted_columns) > 200 else sorted_columns
        param['columns'] = sorted_columns
        return {}

class RFESelectTopNProcess(PNode):

    def __init__(self, n_features_remain, n_features_to_remove, message=None, next=None, previous=None):

        self.n_features_remain = n_features_remain
        self.n_features_to_remove = n_features_to_remove
        self.message = message
        if type(self.message) == type(None):
            self.message = f'{self.__class__.__name__} to {n_features_remain} features'

        super(RFESelectTopNProcess, self).__init__()
        return

    def run(self, df_train, df_test, param, trial):

        ref(df_train, param, trial, message=self.message, key='average_permutation_weight', n_features_remain=self.n_features_remain, n_features_to_remove=self.n_features_to_remove)
        df_trial = pd.DataFrame(trial)
        df_trial_top1 = df_trial[(df_trial['message'] == self.message) & (df_trial['nfeatures'] < 100)].sort_values(by=['val_metric_mean'], ascending=True).head(1)
        columns = df_trial_top1['param'].tolist()[0]['columns']
        score = df_trial_top1['val_metric_mean'].tolist()[0]
        param['columns'] = columns
        return {'score':score}


class RFERemoveUselessFeaturesProcess(PNode):

    def __init__(self, message=None, next=None, previous=None):

        self.message = message
        if type(self.message) == type(None):
            self.message = f'{self.__class__.__name__}'
        super(RFERemoveUselessFeaturesProcess, self).__init__()
        return

    def run(self, df_train, df_test, param, trial, score, **kwargs):

        width_frist_rfe(df_train, param, trial, score=score, message=self.message)
        df_trial = pd.DataFrame(trial)
        columns = df_trial[df_trial['message'] == self.message].sort_values(by=['val_metric_mean'], ascending=True)['param'].tolist()[0]['columns']
        param['columns'] = columns
        return {}


# class OptHyperProcess(PNode):
#
#     def objective(trial):
#         learning_rate = trial.suggest_uniform('learning_rate', .01, .5)
#         feature_fraction = trial.suggest_uniform('feature_fraction', .6, 1)
#         bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.6, 1)
#         min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 200, 800)
#         lambda_l1 = trial.suggest_loguniform('lambda_l1', 1e-6, 1e2)
#         lambda_l2 = trial.suggest_loguniform('lambda_l2', 1e-6, 1e2)
#         max_bin = trial.suggest_int('max_bin', 10, 100)
#         num_leaves = trial.suggest_int('num_leaves', 4, 64)
#         random_state = trial.suggest_int('random_state', 1, 9999)
#
#         args = {
#             'columns': columns,
#             'cv': {
#                 'cls': 'KFold',
#                 'init': {
#                     'n_splits': 5,
#                     'shuffle': True,
#                     'random_state': 42,
#                 },
#             },
#             'scaler': {
#                 'cls': 'StandardScaler',
#                 'init': {},
#                 'fit': {},
#             },
#             'model': {
#                 'cls': 'lgb.LGBMRegressor',
#                 'init': {
#                     'learning_rate': learning_rate,
#                     'feature_fraction': feature_fraction,
#                     'bagging_fraction': bagging_fraction,
#                     'min_data_in_leaf': min_data_in_leaf,
#                     'lambda_l1': lambda_l1,
#                     'lambda_l2': lambda_l2,
#                     'max_bin': max_bin,
#                     'num_leaves': num_leaves,
#                     'random_state': random_state,
#                     'n_jobs': 16
#                 },
#                 'fit': {
#                 },
#             },
#             'metric': 'mean_absolute_error',
#         }
#
#         df_his, df_feature_importances, df_valid_pred, df_test_pred = sk_process(df_train_sample, args,
#                                                                                  'tune hyperparam cv5', trial=mytrial,
#                                                                                  is_output_feature_importance=False,
#                                                                                  trial_level=0)
#         val_metric_mean = np.mean(df_his.valid)
#         return val_metric_mean
#
#     study = optuna.create_study()
#     study.optimize(objective, n_trials=200)
#
#     def __init__(self, next=None, previous=None):
#         super(RFERemoveUselessFeaturesProcess, self).__init__()
#         return
#
#     def run(self, df_train, param, trial, n_features_remain, n_features_to_remove, message=None, **kwargs):
#
#         if type(message) == type(None):
#             message = f'{self.__class__.__name__} to {n_features_remain} features'
#
#         ref(df_train, param, trial, message=message, key='average_permutation_weight', n_features_remain=n_features_remain, n_features_to_remove=n_features_to_remove)
#
#         df_trial = pd.DataFrame(trial)
#         df_trial_top1 = df_trial[(df_trial['message'] == 'rfe') & (df_trial['nfeatures'] < 100)].sort_values(by=['val_metric_mean'], ascending=True).head(1)
#         columns = df_trial_top1['param'].tolist()[0]['columns']
#         score = df_trial_top1['val_metric_mean'].tolist()[0]
#
#         return {'columns':columns, 'score':score}
