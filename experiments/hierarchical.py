import numpy as np
import pandas as pd
import wget
import pickle as pk
from utils.format_regressors import format_regressors as frmt
from mbtr.mbtr import MBT
import matplotlib.pyplot as plt
from utils.benchmark_regressors import MIMO, MISO
from utils.cross_val import cv
from collections import deque
from utils.hierarchical_reconciliation import reconcile_hts

try:
    with open("data/hierarchical_cv_preliminar.npy", "rb") as f:
        all_cv_res = np.load(f,allow_pickle=True).item()
except:
    try:
        nwp_data = pd.read_hdf('data/meteo_data.h5', 'df')
        power_data = pk.load(open("data/power_data.p", "rb"))
    except:
        wget.download('https://zenodo.org/record/3463137/files/power_data.p?download=1', 'data/power_data.p')
        wget.download('https://zenodo.org/record/3463137/files/nwp_data.h5?download=1','data/meteo_data.h5')
        nwp_data = pd.read_hdf('data/meteo_data.h5', 'df')
        power_data = pk.load(open("data/power_data.p", "rb"))

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
    series = power_data['P_mean'].keys()
    n_boosts = 50
    k = 3
    all_cv_res = {}
    for s in series:
        x_pd,y_pd,x,y,t,x_0,y_0,y_hat_persistence = frmt(d=power_data,target_names={'P_mean': [s]},var_names= {'P_mean': [s]},
                                                         pars=format_pars, f=nwp_data,forecasts_names=['temperature', 'ghi_backwards'])
        m = MISO(n_boosts)
        def cv_function(x_tr,y_tr,x_te,y_te):
            m.fit(x_tr,y_tr)
            y_hat_tr = m.predict(x_tr)
            y_hat_te = m.predict(x_te)
            results = {'y_hat_te': y_hat_te,
                       'y_hat_tr': y_hat_tr,
                       'y_tr': y_tr,
                       'y_te': y_te,
                       'x_te': x_te[:,-2:],
                       'x_tr': x_tr[:, -2:]
                       }
            return results
        cv_res = cv(x,y,k,cv_function)
        all_cv_res[s] = cv_res

    np.save('data/hierarchical_cv_preliminar.npy',all_cv_res)

series = list(all_cv_res.keys())
series = deque(series)
series.rotate(7)
print(series)

rec_pars = {'method': 'minT',
           'cov_method': 'shrunk'}

A1l = np.ones((1, len(series) - 7))
A2l = np.kron(np.eye(2), np.ones(int((len(series) - 7) / 2)))
A3l = np.kron(np.eye(4), np.ones(int((len(series) - 7) / 4)))
A = np.vstack([A1l, A2l, A3l])


def hier_scores(x,y,sa):
    groups = ['top','aggs_1','aggs_2','bottoms']
    groups_ids = [[0],[1,2],[3,4,5,6],np.arange(7,y.shape[1])]
    rmse,mape = [{},{}]
    for i,g in enumerate(groups):
        x_g = x[:,groups_ids[i]]
        y_g = y[:, groups_ids[i]]
        rmse[g] = np.mean(np.mean((x_g- y_g)**2)**0.5)
        mape[g] = np.mean(np.mean(np.abs(x_g - y_g)/(np.abs(y_g)+1e-6)))
    scores = pd.DataFrame({'mape':mape,'rmse':rmse,'sa':sa})
    return scores


scores = []
n_max = -1
scores_baseline_all, scores_rec_all, scores_rec_mgb_all = [[],[],[]]
for fold in np.arange(len(all_cv_res[series[0]])):
    scores_baseline,scores_bu, scores_rec, scores_rec_mgb = [pd.DataFrame({}),pd.DataFrame({}), pd.DataFrame({}), pd.DataFrame({})]
    for sa in range(24):
        # collect all the y_hat_tr, y_tr, y_hat_te, y_te
        y_tr,y_te,y_hat_tr,y_hat_te,x_tr,x_te, err_tr, err_te = [[],[],[],[],[],[],[],[]]
        for i,s in enumerate(series):
            y_tr.append(all_cv_res[s][fold]['y_tr'][:n_max, [sa]])
            y_te.append(all_cv_res[s][fold]['y_te'][:n_max, [sa]])
            y_hat_tr.append(all_cv_res[s][fold]['y_hat_tr'][:n_max, [sa]])
            y_hat_te.append(all_cv_res[s][fold]['y_hat_te'][:n_max, [sa]])
            x_tr.append(all_cv_res[s][fold]['x_tr'][:n_max,:])
            x_te.append(all_cv_res[s][fold]['x_te'][:n_max,:])
            err_tr.append(all_cv_res[s][fold]['y_tr'][:n_max, [0]]-all_cv_res[s][fold]['y_hat_tr'][:n_max, [0]])
            err_te.append(all_cv_res[s][fold]['y_te'][:n_max, [0]]-all_cv_res[s][fold]['y_hat_te'][:n_max, [0]])

        y_tr = np.hstack(y_tr)
        y_te = np.hstack(y_te)
        y_hat_tr = np.hstack(y_hat_tr)
        y_hat_te = np.hstack(y_hat_te)
        err_tr = np.hstack(err_tr)
        err_te = np.hstack(err_te)

        # manipulate obs to make available the err at prediction time
        y_tr = y_tr[1:,:]
        y_te = y_te[1:, :]
        y_hat_tr = y_hat_tr[1:, :]
        y_hat_te = y_hat_te[1:, :]
        x_tr[0] = x_tr[0][1:, :]
        x_te[0] = x_te[0][1:, :]
        err_tr = err_tr[:-1, :]
        err_te = err_te[:-1, :]

        # estimate covariance from y_hat_tr, y_tr and reconcile on y_hat_te with previously estimated covariance
        y_rec,precision = reconcile_hts(y_tr, y_hat_tr, y_hat_te, A, rec_pars)
        y_rec = y_rec.T
        # fit a MGB model on y_tr -y_hat_tr_bottom*S.T
        print('start fitting')

        pars = {'n_q': 10,
                'min_leaf': 400,
                'lambda_leaves': 0.1,
                'lambda_weights': 1,
                'early_stopping_rounds': 3,
                'n_boosts': 100,
                'loss_type': 'latent_variable',
                'S': np.vstack([A, np.eye(A.shape[1])]),
                'precision': np.eye(y_hat_tr.shape[1])}

        # predict the MGB model on y_hat_te
        m = MBT(**pars)
        m.fit(np.hstack([y_hat_tr,err_tr,x_tr[0][:,-2:]]),y_tr -y_hat_tr[:,A.shape[0]:]@pars['S'].T, do_plot=True)
        y_rec_mgb = m.predict(np.hstack([y_hat_te,err_te,x_te[0][:,-2:]])) + y_hat_te[:,A.shape[0]:]@pars['S'].T

        y_hat_te_bu = y_hat_te[:,A.shape[0]:]@pars['S'].T

        scores_baseline = pd.concat([scores_baseline, hier_scores(y_te, y_hat_te, sa)], axis=0)
        scores_bu = pd.concat([scores_bu, hier_scores(y_te, y_hat_te_bu, sa)], axis=0)
        scores_rec = pd.concat([scores_rec, hier_scores(y_te, y_rec, sa)], axis=0)
        scores_rec_mgb = pd.concat([scores_rec_mgb,hier_scores(y_te, y_rec_mgb, sa)], axis=0)
        print(scores_bu)
        print(scores_rec)
        print(scores_rec_mgb)

        N = 1000
        k = 0
        plt.figure()
        plt.plot(y_te[0:N,k],'--',label='y_te')
        plt.plot(y_hat_te[0:N,k],linewidth=2,label='y_hat_te')
        plt.plot(y_rec[0:N,k],label='rec')
        plt.plot(y_rec_mgb[0:N,k],'--.',label='mgb')
        plt.legend()

    scores_f = {'base': scores_baseline,
                'bu':scores_bu,
                  'rec': scores_rec,
                  'mgb': scores_rec_mgb}

    scores.append(scores_f)

    plt.close('all')


np.save('data/hierarchical_scores.npy',scores)


