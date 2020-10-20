import numpy as np
import pandas as pd
import lightgbm as lgb
import tqdm

class MISO:
    def __init__(self, n_estimators, lgb_pars = None):
        self.n_estimators = n_estimators
        if lgb_pars is None:
            self.lgb_pars = {"objective": "regression",
                             "max_depth": 20,
                             "num_leaves": 100,
                             "learning_rate": 0.1,
                             "verbose": -1,
                             "metric": "l2",
                             "min_data": 4,
                             "num_threads":4}
        else:
            self.lgb_pars = lgb_pars
        self.m = []

    def fit(self, x, y):
        self.m = []
        for i in tqdm.tqdm(range(y.shape[1])):
            lgb_train = lgb.Dataset(x, y[:, i].ravel())
            self.m.append(lgb.train(self.lgb_pars, lgb_train, num_boost_round=self.n_estimators))
        return self

    def predict(self,x):
        y = []
        for m in self.m:
            y.append(m.predict(x).reshape(-1,1))
        y = np.hstack(y)
        return y


class MIMO:
    def __init__(self, n_estimators, lgb_pars=None):
        self.n_estimators = n_estimators
        if lgb_pars is None:
            self.lgb_pars = {"objective": "regression",
                             "max_depth": 20,
                             "num_leaves": 100,
                             "learning_rate": 0.1,
                             "verbose": -1,
                             "metric": "l2",
                             "min_data": 4,
                             "num_threads":4}
        else:
            self.lgb_pars = lgb_pars
        self.m = []
        self.n_targets = 0

    def fit(self, x, y):
        self.n_targets = y.shape[1]
        x_all = []
        for i in range(self.n_targets):
            x_i = np.hstack([x,i*np.ones((len(x),1))])
            x_all.append(x_i.astype(np.float32))
        x_all = pd.DataFrame(np.vstack(x_all),columns=np.arange(x.shape[1]+1),dtype=np.float32)
        y = y.ravel(order='F')

        lgb_train = lgb.Dataset(x_all, y)
        self.m = lgb.train(self.lgb_pars, lgb_train, num_boost_round=self.n_estimators, categorical_feature=[x.shape[1]+1])

    def predict(self, x):
        y = []
        for i in range(self.n_targets):
            x_i = np.hstack([x,i*np.ones((len(x),1))])
            x_all = pd.DataFrame(np.vstack(x_i), columns=np.arange(x.shape[1] + 1)).astype(np.float32)
            y.append(self.m.predict(x_all).reshape(-1,1))
        y = np.hstack(y)
        return y
