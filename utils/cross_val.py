from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import sharedmem
from time import time
from mbtr.mbtr import MBT


def cv(x,y, k, func, x_build=None):
    """
    Perform a forward CV. In each fold we keep the training and test proportion fixed to 3/4, 1/4.
    In this way, the overall test data ratio is = 1 / (k + 3)

    k = 3. Test ratio = 1/6
    |---|---|---|---|---|---|
    |---|---|---|###|
        |---|---|---|###|
            |---|---|---|###|

    k = 4. Test ratio = 1/7
    |---|---|---|---|---|---|---|
    |---|---|---|###|
        |---|---|---|###|
            |---|---|---|###|
                |---|---|---|###|

    :param x: covariate matrix
    :param y: target matrix
    :param k: folds
    :param func:
    :return:
    """
    n_obs = y.shape[0]
    n_te = np.floor(n_obs / (k+3)).astype(int)
    results = []
    for i in range(k):
        tr_win = np.arange(n_te*3) + i * n_te
        te_win = tr_win[-1] + np.arange(n_te)
        print_fold(tr_win,te_win,n_obs)
        x_tr, y_tr = [x[tr_win, :], y[tr_win, :]]
        x_te, y_te = [x[te_win, :], y[te_win, :]]
        if x_build is not None:
            x_build_tr, x_build_te = [x_build[tr_win, :], x_build[te_win, :]]
            r = func(x_tr, y_tr, x_te, y_te, x_build_tr, x_build_te)
        else:
            r = func(x_tr,y_tr,x_te,y_te)


        results.append(r)
    return results


def cv_partial(i, x, y, k, func, x_build=None):
    """
    Perform a forward CV. In each fold we keep the training and test proportion fixed to 3/4, 1/4.
    In this way, the overall test data ratio is = 1 / (k + 3)

    k = 3. Test ratio = 1/6
    |---|---|---|---|---|---|
    |---|---|---|###|
        |---|---|---|###|
            |---|---|---|###|

    k = 4. Test ratio = 1/7
    |---|---|---|---|---|---|---|
    |---|---|---|###|
        |---|---|---|###|
            |---|---|---|###|
                |---|---|---|###|

    :param x: covariate matrix
    :param y: target matrix
    :param k: folds
    :param func:
    :return:
    """
    n_obs = y.shape[0]
    n_te = np.floor(n_obs / (k + 3)).astype(int)
    tr_win = np.arange(n_te * 3) + i * n_te
    te_win = tr_win[-1] + np.arange(n_te)
    print_fold(tr_win, te_win, n_obs)
    x_tr, y_tr = [x[tr_win, :], y[tr_win, :]]
    x_te, y_te = [x[te_win, :], y[te_win, :]]
    if x_build is not None:
        x_build_tr, x_build_te = [x_build[tr_win, :], x_build[te_win, :]]
        r = func(x_tr, y_tr, x_te, y_te, x_build_tr, x_build_te)
    else:
        r = func(x_tr, y_tr, x_te, y_te)

    #r = func(x_tr, y_tr, x_te, y_te)

    return r


def parallel_cv(x,y, k, func, x_build=None):
    results = mapper(cv_partial, np.arange(k), x, y, k=k, func=func, x_build=x_build)
    return results


def mapper(f,pars,*argv,**kwarg):
    # create a shared object from X
    x_s = [sharedmem.copy(x) for x in argv]
    # parallel process over shared object
    t0 = time()
    with mp.Pool(3) as pool:
        res = pool.starmap_async(f,[(p,*x_s,*list(kwarg.values())) for p in pars])
        a = res.get()
    pool.close()
    pool.join()
    return a


def print_fold(tr,te, N):
    display_len = 100
    s, e, n = (np.array([tr[0], len(tr), N - (tr[0] + len(tr))]) * display_len / N).astype(int)
    tr_str = '-' * s + '#' * e + '-' * n
    s, e, n = (np.array([te[0], len(te), N - (te[0] + len(te))]) * display_len / N).astype(int)
    te_str = '-' * s + '#' * e + '-' * n
    print('train: {}'.format(tr_str))
    print('test:  {}'.format(te_str))


def quantile_scores(q_hat: np.ndarray, y: np.ndarray, alphas):
    s_k_alpha, reliability_alpha = [[],[]]

    for a, alpha in enumerate(alphas):
        err = q_hat[:, a, :] - y
        I = np.asanyarray(err > 0, dtype=int)
        s_k = (I - alpha) * err
        s_k_alpha.append(np.mean(s_k, axis=0))
        reliability_alpha.append(np.mean(I, axis=0))

    s_k_alpha = np.vstack(s_k_alpha)
    reliability_alpha = np.vstack(reliability_alpha)

    crps_t = np.sum(s_k_alpha, axis=0)
    crps_mean = np.mean(crps_t)

    xcross = []
    for t in range(q_hat.shape[2]):
        q_hat_t = q_hat[:,:,t]
        crosses = 0
        for a in range(len(alphas)-1):
            crosses += np.mean(q_hat_t[:,[a]] > q_hat_t[:,a+1:])
        xcross.append(crosses)
    xcross = np.hstack(xcross)
    scores = {'reliability': reliability_alpha,
              'skill': s_k_alpha,
              'crps_t': crps_t,
              'crps_mean': crps_mean,
              'crosses': xcross}
    return scores


def get_cv_results(res):
    cv_res = mediate_nested_dicts_list(res)
    return cv_res


def mediate_nested_dicts_list(list):
    r = deepcopy(list[0])
    r = clean_dict(r)
    paths = np.array(get_paths(r)).flatten()
    for p in paths:
        for l in list:
            exec('r{} = r{} + l{}/{}'.format(p,p,p,len(list)))

    return r


def get_paths(d):
    paths = []
    if type(d) == dict:
        for k, v in d.items():
            if type(v) == dict:
                if v=={}:
                    paths = paths + [strfy(k)]
                else:
                    paths_v = get_paths(v)
                    paths = paths + [strfy(k,p) for p in paths_v]
            else:
                paths = paths + [strfy(k)]
    else:
        return
    return paths


def strfy(k, p=None):
    if p is not None:
        if type(k) == str:
            return '["{}"]{}'.format(k, p)
        else:
            return '[{}]{}'.format(k, p)
    if type(k) == str:
        return '["{}"]'.format(k)
    else:
        return '[{}]'.format(k)


def clean_dict(r):
    if type(r) == dict:
        for k, v in r.items():
            if type(v) == dict:
                r[k] = clean_dict(v)
            else:
                if (type(r[k]) == np.ndarray) or (np.isscalar(r[k])) or (type(r[k]) == list):
                    r[k] = r[k] * 0
                elif type(r[k]) == pd.DataFrame:
                    r[k] = pd.DataFrame(0,columns=r[k].columns, index=r[k].index)
    return r


def get_scores(y_hat,y_te,y_dummy):
    scores = {}
    err = y_hat-y_te
    err_dummy = y_dummy - y_te
    filter_idx = np.abs(y_te)>1e-2
    scores['mape'] = np.mean(np.abs(err[filter_idx])/np.abs(y_te[filter_idx]))
    scores['rmse'] = np.mean(err**2)**0.5
    dummy_mape = np.mean(np.abs(err_dummy[filter_idx]) / np.abs(y_te[filter_idx]))
    dummy_rmse = np.mean(err_dummy ** 2) ** 0.5

    scores['norm_mape'] = scores['mape'] / dummy_mape
    scores['norm_rmse'] = scores['rmse'] / dummy_rmse

    return scores


def truncnorm_rvs_recursive(n,sigmas):
    q = np.random.normal(0, 1, size=n).ravel()
    lower_clip = -sigmas
    upper_clip = sigmas
    if np.any(q < lower_clip):
        q[q < lower_clip] = truncnorm_rvs_recursive(np.sum(q < lower_clip), sigmas)
    if np.any(q > upper_clip):
        q[q > upper_clip] = truncnorm_rvs_recursive(np.sum(q > upper_clip), sigmas)
    return q


def partial_vsc(x_lingb_tr,y_tr_d,x_tr_d,x_lingb_te, x_te_d,y_te_d,y_mean, model):
    n_nodes = int(x_tr_d.shape[1]/2)
    scores = {}
    for n_control in [1,2,4,8,12,16]:
        control_groups = np.vstack([np.random.permutation(n_nodes)[0:n_control] for _ in range(10)])
        scores_partial,scores_noise_partial = [[],[]]
        for c_group in control_groups:
            x_tr_partial = np.hstack([x_tr_d[:, [i]] for i in range(2 * n_nodes) if
                            i not in np.hstack([c_group, c_group + n_nodes])])

            x_te_partial = np.hstack([x_te_d[:, [i]] for i in range(2 * n_nodes) if
                            i not in np.hstack([c_group, c_group + n_nodes])])

            x_tree_tr = np.hstack([x_lingb_tr,x_tr_partial])
            x_tree_te = np.hstack([x_lingb_te,x_te_partial])

            model.re_init()
            model_fitted = model.fit(x_tree_tr, y_tr_d, x_lr=x_tr_d, do_plot=True, do_refit=False)
            y_hat = model_fitted.predict(x_tree_te, x_lr=x_te_d)
            sc = get_scores(y_hat, y_te_d, y_mean)
            plt.figure()
            plt.scatter(y_te_d,y_hat, marker='.', alpha=0.2)
            plt.pause(0.1)
            print('group {}'.format(c_group))
            print(sc)
            scores_partial.append(sc)
            scores_noise = {}
            for k,k_noise in enumerate(range(10)):
                noise = []
                for v in x_tree_te.T:
                    noise.append(truncnorm_rvs_recursive(x_te_d.shape[0],3).reshape(-1,1)*(np.mean(np.abs(v))*(k_noise/10)))
                y_hat_noise = model_fitted.predict(x_tree_te + np.hstack(noise), x_lr=x_te_d)
                sc_n = get_scores(y_hat_noise, y_te_d, y_mean)
                scores_noise['noise:{}'.format(k)] = sc_n
            scores_noise_partial.append(scores_noise)
        scores_noise_partial_cv = mediate_nested_dicts_list(scores_noise_partial)
        scores_partial_cv = mediate_nested_dicts_list(scores_partial)
        print('cv results n_control={}'.format(n_control))
        print(scores_partial_cv)
        all_scores = {'scores_partial_cv':scores_partial_cv,
                     'scores_noise_partial_cv':scores_noise_partial_cv}
        scores['n_control_{}'.format(n_control)] = all_scores
    plt.close('all')

    return scores


def noise_vsc(x_lingb_tr,y_tr_d,x_tr_d,x_lingb_te, x_te_d,y_te_d,y_mean, model):
    x_tree_tr = x_lingb_tr
    x_tree_te = x_lingb_te
    model = MBT(**{'n_boosts':model.n_boosts, 'early_stopping_rounds':model.early_stopping_rounds, **model.tree_pars})
    model_fitted = model.fit(x_tree_tr, y_tr_d, x_lr=x_tr_d, do_plot=True)
    y_hat = model_fitted.predict(x_tree_te, x_lr=x_te_d)
    sc = get_scores(y_hat, y_te_d, y_mean)
    plt.figure()
    plt.scatter(y_te_d,y_hat, marker='.', alpha=0.2)
    plt.pause(0.1)
    print(sc)
    scores_noise = {}
    x_scores = {'noise:{}'.format(k):{} for k in np.arange(10)}
    for k,k_noise in enumerate(range(10)):
        # add noise only to the first x_tree_te variable (the PCC power) The other are deterministic
        noise = []
        for j in range(x_tree_te.shape[1]-2):
            noise_j = truncnorm_rvs_recursive(x_te_d.shape[0],3).reshape(-1,1)*(np.mean(np.abs(x_tree_te[0,:]))*(k_noise/50))
            noise.append(noise_j.reshape(-1,1))
        noise = np.hstack(noise)

        x_hat = x_tree_te + np.hstack([noise,np.zeros((len(noise),2))])
        x_scores['noise:{}'.format(k)]['mape'] = np.mean(np.abs(x_hat[:,0]-x_tree_te[:,0])/np.abs(x_tree_te[:,0] + 1e-2),axis=0)
        x_scores['noise:{}'.format(k)]['rmse'] = np.mean((x_hat[:,0]-x_tree_te[:,0])**2,axis=0)**0.5
        y_hat_noise = model_fitted.predict(x_hat, x_lr=x_te_d)
        sc_n = get_scores(y_hat_noise, y_te_d, y_mean)
        scores_noise['noise:{}'.format(k)] = sc_n
    all_scores = {**sc,
                  'scores_noise_partial_cv':scores_noise,
                  'x_scores': x_scores}
    return all_scores
