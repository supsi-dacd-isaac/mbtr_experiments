import numpy as np
from sklearn.covariance import GraphicalLassoCV, ShrunkCovariance


def reconcile_hts(y_tr: np.ndarray, y_hat_tr: np.ndarray, y_hat_te: np.ndarray, A: np.ndarray, pars: dict):
    """
    Reconcile hierarchical time series. Reconcile only one step ahead, in order not to overflow RAM
    :param y_tr: np.ndarray (n_obs, n_b+n_u), ground truth
    :param y_hat_tr: np.ndarray (n_obs, n_b+n_u), forecasted values for the training set. These and y_tr will be used
    to estimated the error covariance
    :param y_hat_te: np.ndarray (n_obs, n_b+n_u), forecasted values for the test set. These will be used for the
    reconciliation
    :param A: np.ndarray matrix (n_b+n_u,n_b), such as y_bu = A*y_b
    :param pars: dict containig {'cov_method': covariance estiamtion method in ['glasso','shrunk'],
                                 'method': reconciliation method, in ['minT','bayes']}
    :return: y_rec reconciled forecasts (n_obs,n_b+n_u)
    """

    # get forecast errors for the bottom and upper time series
    err = y_tr-y_hat_tr
    n_obs,n_all = err.shape
    n_b = A.shape[1]
    n_u = n_all - n_b
    u_idx = np.arange(n_u)
    b_idx = np.arange(n_b) + n_u
    err_b = err[:, b_idx]   # errors for the bottom time series
    err_u = err[:, u_idx]   # errors for the upper time series

    I = np.eye(n_b)
    S = np.vstack([A,I])

    y_rec = None
    # estimate covariances
    if pars['method'] == 'dummy':
        precision = np.eye(n_b+n_u)
        P = np.linalg.inv(S.T @ precision @ S) @ (S.T @ precision)
        y_rec = S @ P @ y_hat_te.T
    elif pars['method'] == 'minT':
        cov, precision = estimate_covariance(err,method = pars['cov_method'])
        P = np.linalg.inv(S.T@ precision @ S) @ (S.T @ precision)
        y_rec = S @ P @ y_hat_te.T
    elif pars['method'] == 'bayes':
        cov_b, precision_b = estimate_covariance(err_b,method = pars['cov_method'])
        if err_u.shape[1]<2:
            cov_u = np.std(err_u)
        else:
            cov_u, precision_u = estimate_covariance(err_u,method = pars['cov_method'])
        y_b = y_hat_te[:, n_u:]
        y_u = y_hat_te[:, :n_u]
        G = cov_b@ A.T @ np.linalg.inv(cov_u + A @ cov_b @ A.T)
        y_b_rec = y_b.T + G @ (y_u.T - (A @ y_b.T))
        y_rec = S @ y_b_rec

        # get posterior covariance of bottoms
        cov_b_post = cov_b -G.dot(cov_u + A.dot(cov_b.dot(A.T))).dot(G.T)
        cov_post = S.dot(cov_b_post.dot(S.T))
    else:
        ValueError('method is not in [minT,bayes]')

    return y_rec, precision


def estimate_covariance(x,method):
    """
    Covariance estimator wrapper
    :param x:
    :param method:
    :return:
    """
    cov = None
    if method == 'shrunk':
        cov = ShrunkCovariance().fit(x)
    elif method == 'glasso':
        cov = GraphicalLassoCV(cv=5,alphas=10,n_refinements=10).fit(x)
    else:
        ValueError('Covariance method not in [shrunk,glasso]')

    return cov.covariance_, cov.precision_