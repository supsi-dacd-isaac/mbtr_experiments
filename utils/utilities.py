import numpy as np
import pandas as pd
from typing import Sized, Union


def list_df_to_matrix(D: pd.DataFrame):
    X = []
    for k in D.keys():
        v = np.vstack(D[k].values)
        X.append(v)
    X = np.hstack(X)
    return X, D.index


def matrix_to_list_df(X: np.ndarray, index: Union[np.ndarray, pd.DatetimeIndex], name: str = '0') -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df.loc[:, name] = list(X)
    return df


def reduce_matrix(x: np.ndarray, reduction_bins: list) -> np.ndarray:
    """
    Reduce matrix x through averaging columns based on the reduction bins values
    :param x: numpy ndarray, size (n_obs,h)
    :param reduction_bins: list of k aggregation steps, which must sum to h
    :return: numpy ndarray, reduced matrix size (n_obs, k)
    """
    n_obs, h = x.shape

    # check reduction_bins
    if not sum(reduction_bins) == h:
        raise ValueError('Bins in reduction bins must sum to the second dimension of x')

    x_red = []
    i = 0
    for r in reduction_bins:
        x_r = np.mean(x[:, i:i + r], axis=1)
        x_red.append(x_r.reshape(x.shape[0], -1))
        i += r

    return np.hstack(x_red)


def aggregated_reduction(x: np.ndarray, reduction_bins: list) -> np.ndarray:
    """
    Reduce matrix x through averaging columns based on the reduction bins values
    :param x: numpy ndarray, size (n_obs,h)
    :param reduction_bins: indeces of columns to aggregate, expressed in an accumarray fashion
    :return: numpy ndarray, reduced matrix size (len(unique(reduction_bins)), k)
    """
    n_obs, h = x.shape
    reduction_bins = np.array(reduction_bins).reshape(-1, 1)
    # check reduction_bins
    if len(reduction_bins) > h:
        raise ValueError('Len of reduction bins must be lower than the second dimension of x')

    classes = np.unique(reduction_bins)
    x_red = []
    for c in classes:
        filter = reduction_bins == c
        vals = np.mean(x[:, filter.ravel()], 1).reshape(-1, 1)
        x_red.append(vals)
    x_red = np.hstack(x_red)

    return x_red


def reduce_dataset(d: pd.DataFrame, reduction_pars: dict):
    """
    Reduces the data contained in a pandas DataFrame
    :param d: pandas DataFrame. Each column contains lists of numbers
    :param reduction_pars: dict containing 'type' and 'values'. 'type' describes the type of reduction performed on the
    lists in d.
    :return:
    """
    p = pd.DataFrame(index=d.index)
    for k in d:
        if reduction_pars['type'] == 'bins':
            p[k] = list(reduce_matrix(np.vstack(d[k].values), reduction_pars['values']))
        if reduction_pars['type'] == 'aggregated_selection':
            if np.all(reduction_pars['values'] == np.arange(len(d[k][0]))):
                p[k] = d[k]
            else:
                p[k] = list(aggregated_reduction(np.vstack(d[k].values), reduction_pars['values']))
        if reduction_pars['type'] == 'min':
            p[k] = np.min(np.vstack(d[k].values), axis=1)
        if reduction_pars['type'] == 'max':
            p[k] = np.max(np.vstack(d[k].values), axis=1)
        if reduction_pars['type'] == 'mean':
            p[k] = np.mean(np.vstack(d[k].values), axis=1)

    return p


def hankel(x, n, generate_me=None):
    if generate_me is None:
        generate_me = np.arange(n)
    x = x.ravel()
    m = len(x) - n + 1
    w = np.arange(m)
    h = []
    for g in generate_me:
        h.append(x[w + g].reshape(-1, 1))
    return np.hstack(h)


def hankelize_vector(x: np.ndarray, d: int, generate_me=None) -> np.ndarray:
    '''
    Return the Hankel matrix for vector x, with embedding dimension d
    :param x: nump ndarray, length n_obs
    :param d: int, embedding dimension
    :return: numpy ndarray, dimension (n_obs-d+1,d)
    '''

    h = hankel(x, d, generate_me)

    return h[0:len(x) - d + 1, :]


def hankelize_matrix(x: np.ndarray, d: int, generate_me=None) -> np.ndarray:
    '''
    Return the Hankel matrix for vector x, with embedding dimension d
    :param x: nump ndarray, length n_obs
    :param d: int, embedding dimension
    :return: numpy ndarray, dimension (n_obs-d+1,d)
    '''

    H = []
    for i in np.arange(x.shape[1]):
        h = hankelize_vector(x[:, [i]], d, generate_me)
        H.append(h)

    H = np.hstack(H)


    return H