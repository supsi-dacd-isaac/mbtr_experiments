import numpy as np
import pandas as pd
import wget
import pickle as pk
from utils.format_regressors import format_regressors as frmt
from mbtr.mbtr import MBT
import matplotlib.pyplot as plt
from utils.benchmark_regressors import MIMO, MISO
from utils.cross_val import cv, quantile_scores, parallel_cv
from time import time
from copy import deepcopy

try:
    nwp_data = pd.read_hdf('data/meteo_data.h5', 'df')
    power_data = pk.load(open("data/power_data.p", "rb"))
except:
    wget.download('https://zenodo.org/record/3463137/files/power_data.p?download=1', 'data/power_data.p')
    wget.download('https://zenodo.org/record/3463137/files/nwp_data.h5?download=1', 'data/meteo_data.h5')
    nwp_data = pd.read_hdf('data/meteo_data.h5', 'df')
    power_data = pk.load(open("data/power_data.p", "rb"))


aggregations = np.tile(np.arange(24).reshape(-1,1),6).ravel()
format_pars = {'h_f':144,
               'h_b':144,
               'x_reduction': {'type': 'aggregated_selection',
                               'values': aggregations},
               'y_reduction': {'type': 'aggregated_selection',
                               'values': aggregations},
               'f_reduction': None,
               'hour': True,
               'week_day': True,
               'vacation': False
               }

n_quantiles = 11
n_boost = 100
alphas = np.linspace(1/n_quantiles, 1-1/n_quantiles, n_quantiles)

pars = {'n_q': 20,
        'min_leaf': 300,
        'early_stopping_rounds': 6,
        'n_boosts': n_boost,
        'loss_type': 'quantile',
        'alphas': alphas,
        'lambda_leaves': 0.01,
        'lambda_weights': 1,
        'shift': 1
        }

pars_squared = deepcopy(pars)
pars_squared['loss_type'] = 'quadratic_quantile'

lgb_pars = {"objective": "quantile",
            "metric": "quantile",
             "num_leaves": n_boost,
             "learning_rate": 0.1,
             "verbose": -1,
             "min_data": 4,
            "num_threads": 4}


def cv_function(x_tr,y_tr,x_te,y_te,pars,do_refit):
    n_t = y_tr.shape[1]
    q_hat, q_hat_mgb = [[],[]]
    pars['refit'] = do_refit
    for t in range(n_t):
        # fit LGBM model
        q_hat_t = []
        for alpha in alphas:
            lgb_pars['alpha'] = alpha
            m = MISO(n_boost, lgb_pars)
            m.fit(x_tr, y_tr[:,[t]])
            q_hat_t.append(m.predict(x_te))
        q_hat_t = np.hstack(q_hat_t)
        q_hat.append(q_hat_t)

        # fit MBT model
        m = MBT(**pars)
        m.fit(x_tr, y_tr[:, [t]], do_plot=True)
        q_hat_t = m.predict(x_te)

        q_hat_mgb.append(q_hat_t)
        results_miso_t = quantile_scores(np.expand_dims(q_hat[-1],2), y_te[:,[t]], alphas)
        results_mgb_t = quantile_scores(np.expand_dims(q_hat_mgb[-1],2), y_te[:, [t]], alphas)
        print('crsp miso {:0.2e}, mbg {:0.2e}'.format(results_miso_t['crps_mean'],results_mgb_t['crps_mean']))
    q_hat = np.dstack(q_hat)
    q_hat_mgb = np.dstack(q_hat_mgb)

    results_miso = quantile_scores(q_hat, y_te, alphas)
    results_mgb = quantile_scores(q_hat_mgb, y_te, alphas)
    results = {'miso': results_miso,
               'mgb': results_mgb}
    plt.close('all')
    return results


def cv_function_no_refit(x_tr,y_tr,x_te,y_te):
    return cv_function(x_tr,y_tr,x_te,y_te,deepcopy(pars),do_refit=False)


def cv_function_refit(x_tr,y_tr,x_te,y_te):
    return cv_function(x_tr,y_tr,x_te,y_te,deepcopy(pars),do_refit=True)


def cv_function_squared_refit(x_tr,y_tr,x_te,y_te):
    return cv_function(x_tr,y_tr,x_te,y_te,deepcopy(pars_squared),do_refit=True)


k = 3
series = list(power_data['P_mean'].keys())
cv_res_refit,cv_res_squared_refit,cv_res_no_refit = [{},{},{}]

for s in ['all']:
    aggregations = np.tile(np.arange(24).reshape(-1,1),6).ravel()
    format_pars = {'h_f':144,
                   'h_b':144,
                   'x_reduction': {'type':'aggregated_selection',
                                   'values':aggregations},
                   'y_reduction': {'type': 'aggregated_selection',
                                   'values': aggregations},
                   'f_reduction': None,
                   'hour':True,
                   'week_day':True,
                   'vacation':False
                   }
    x_pd,y_pd,x,y,t,x_0,y_0,y_hat_persistence = frmt(d=power_data,target_names={'P_mean': [s]},var_names= {'P_mean': [s]},
                                                     pars=format_pars, f=nwp_data,forecasts_names=['temperature', 'ghi_backwards'])

    x = x[np.asanyarray(np.kron(np.ones(int(len(x) / 6)), [1,0,0,0,0,0]),dtype=bool),:]
    y = y[np.asanyarray(np.kron(np.ones(int(len(y) / 6)), [1, 0, 0, 0,0,0]), dtype=bool), :]


    t_0 = time()
    cv_res_h_0 = parallel_cv(x, y, k, cv_function_squared_refit)
    cv_res_squared_refit[s] = deepcopy(cv_res_h_0)

    cv_res_h_1 = parallel_cv(x, y, k, cv_function_refit)
    cv_res_refit[s] = deepcopy(cv_res_h_1)

    cv_res_h_2 = parallel_cv(x, y, k, cv_function_no_refit)
    cv_res_no_refit[s] = deepcopy(cv_res_h_2)

    cv_res_all = {'no_refit':cv_res_no_refit,
                  'refit':cv_res_refit,
                  'suqared_refit':cv_res_squared_refit}
    print('CV for series {} done in {:0.2e} mins'.format(s, (time()-t_0)/60))
    np.save('data/cv_res_quantiles_all_squared_partial.npy', cv_res_all)
