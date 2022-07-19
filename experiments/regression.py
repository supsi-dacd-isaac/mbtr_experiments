import numpy as np
import pandas as pd
import wget
import pickle as pk
from utils.format_regressors import format_regressors as frmt
from mbtr.mbtr import MBT
from utils.benchmark_regressors import MIMO, MISO
from utils.cross_val import cv, parallel_cv
from copy import deepcopy
from time import time

try:
    data = pd.read_csv('data/M4.csv', index_col=0)
except:
    wget.download('https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv',
                  'data/M4.csv')
    data = pd.read_csv('data/M4.csv', index_col=0)

# filter out NaNs
data = data.T
data = data.loc[data.isna().sum(axis=1)==0, :]
data.reset_index(inplace=True, drop=True)
data.index = pd.date_range(start='2000-01-01', freq='1d', periods=len(data))
data = {'data': data}

aggregations_f = np.arange(24)
aggregations_b = np.arange(48)
format_pars = {'h_f': 24,
               'h_b': 48,
               'x_reduction': {'type': 'aggregated_selection',
                               'values': aggregations_b},
               'y_reduction': {'type': 'aggregated_selection',
                               'values': aggregations_f},
               'f_reduction': None,
               'hour': True,
               'week_day': True,
               'vacation': False
               }

# ----------------------------------------------------
# CV using MSE loss
# ----------------------------------------------------
# set pars
n_boosts = 100
n_leaves = 100
lgb_pars = {"objective": "regression",
            "learning_rate": 0.1,
            "verbose": -1,
            "metric": "l2",
            "min_data": n_leaves,
            "num_threads": 4}

mbt_pars_0 = {"n_boosts": n_boosts,
              "n_q": 10,
              "early_stopping_rounds": 7,
              "min_leaf": n_leaves,
              "loss_type": 'mse',
              "lambda_leaves": 0.1,
              "lambda_weights": 0.1
              }


def cv_fun(x_tr,y_tr,x_te,y_te):
    names = ['mimo','miso']
    models = [MIMO(n_boosts, lgb_pars),MISO(n_boosts, lgb_pars)]
    y_hat = {}
    for i, m in enumerate(models):
        m.fit(x_tr, y_tr)
        y_hat[names[i]] = m.predict(x_te)

    rmse_base, mape_base = [{}, {}]
    for i,n in enumerate(names):
        rmse_base[n] = np.mean(np.mean((y_hat[n] - y_te)**2, axis=0)**0.5)
        mape_base[n] = np.mean(np.mean(np.abs(y_hat[n] - y_te)/(np.abs(y_te)+1e-3), axis=0))
        print('{} rmse: {:0.2e}, mape: {:0.2e}'.format(n,rmse_base[n],mape_base[n]))

    m = MBT(**deepcopy(mbt_pars_0))
    m.fit(x_tr, y_tr, do_plot=False)
    y_hat_mbt = m.predict(x_te)
    rmse = np.mean(np.mean((y_hat_mbt - y_te) ** 2, axis=0) ** 0.5)
    mape = np.mean(np.mean(np.abs(y_hat_mbt - y_te) / (np.abs(y_te) + 1e-3), axis=0))
    print('MAPE w.r.t. {}: {:0.2e}'.format(names[0], mape/mape_base[names[0]]))
    results = {'mape': mape,
               'rmse': rmse,
               'mape_mimo': mape_base['mimo'],
               'mape_miso': mape_base['miso'],
               'rmse_mimo': rmse_base['mimo'],
               'rmse_miso': rmse_base['miso']
               }
    return results


k = 3
series = list(data['data'].keys())[:50]
cv_res = {}
for s in series:
    x_pd, y_pd, x, y, t, x_0, y_0, y_hat_persistence = frmt(d=data, target_names={
        'data': [s]}, var_names={ 'data': [s]}, pars=format_pars)

    t_0 = time()

    cv_res_h = parallel_cv(x, y, k, cv_fun)
    cv_res[s] = cv_res_h
    print('CV for series {} done in {:0.2e} mins'.format(s, (time()-t_0)/60))
    np.save('data/cv_res_m4.npy', cv_res)

# ----------------------------------------------------
# CV using Fourier loss
# ----------------------------------------------------

mbt_pars_0 = {"n_boosts": n_boosts,
              "n_q": 10,
              "early_stopping_rounds": 7,
              "min_leaf": n_leaves,
              "loss_type": 'fourier',
              "n_harmonics": 30,
              "lambda_leaves": 0.1,
              "lambda_weights": 0.1
              }


def cv_fun(x_tr,y_tr,x_te,y_te):
    names = ['mimo','miso']
    models = [MIMO(n_boosts, lgb_pars),MISO(n_boosts, lgb_pars)]
    y_hat = {}
    for i, m in enumerate(models):
        m.fit(x_tr, y_tr)
        y_hat[names[i]] = m.predict(x_te)

    rmse_base, mape_base = [{}, {}]
    for i,n in enumerate(names):
        rmse_base[n] = np.mean(np.mean((y_hat[n] - y_te)**2, axis=0)**0.5)
        mape_base[n] = np.mean(np.mean(np.abs(y_hat[n] - y_te)/(np.abs(y_te)+1e-3), axis=0))
        print('{} rmse: {:0.2e}, mape: {:0.2e}'.format(n,rmse_base[n],mape_base[n]))

    mape, rmse = [{}, {}]
    for n_h in np.arange(4, 24, 2):
        mbt_pars = deepcopy(mbt_pars_0)
        mbt_pars['n_harmonics'] = n_h

        m = MBT(**mbt_pars)
        m.fit(x_tr, y_tr, do_plot=False)
        y_hat_mbt = m.predict(x_te)

        rmse[n_h] = np.mean(np.mean((y_hat_mbt - y_te)**2, axis=0)**0.5)
        mape[n_h] = np.mean(np.mean(np.abs(y_hat_mbt - y_te)/(np.abs(y_te)+1e-3), axis=0))
        print('rmse: {:0.4e}, mape: {:0.4e}, n_h {}'.format(rmse[n_h],mape[n_h], n_h))


    results = {'mape':mape,
               'rmse':rmse,
               'mape_mimo': mape_base['mimo'],
               'mape_miso':mape_base['miso'],
               'rmse_mimo': rmse_base['mimo'],
               'rmse_miso':rmse_base['miso']
               }
    return results


# ----------------------------------------------------
# CV on all the time series
# ----------------------------------------------------
k = 3
series = list(data['data'].keys())
cv_res = {}
for s in series:
    x_pd, y_pd, x, y, t, x_0, y_0, y_hat_persistence = frmt(d=data, target_names={
        'data': [s]}, var_names={ 'data': [s]}, pars=format_pars)

    t_0 = time()

    cv_res_h = parallel_cv(x, y, k, cv_fun)
    cv_res[s] = cv_res_h
    print('CV for series {} done in {:0.2e} mins'.format(s, (time()-t_0)/60))
    np.save('data/cv_res_m4_fourier.npy', cv_res)

