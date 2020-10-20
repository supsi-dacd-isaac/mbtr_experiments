import numpy as np
import pandas as pd
import utils.utilities as ut
from typing import Union


def format_regressors(d: dict, target_names: dict, var_names: dict, pars: dict, f: pd.DataFrame = None,
                      g: pd.DataFrame = None, forecasts_names: list = None, global_names: list = None) -> [np.ndarray]:
    """
    Prepare a numpy ndarray matrix of regressors and targets for the forecasters
    :param d: list of pandas.Dataframe instances, whose columns are indexed with the nodes names
    :param target_names: dict of variables of d which are handled like target variables.
             Each key representing the name of a variable, values are list with the names of the nodes for
             which this variable must be retrieved
    :param var_names: dict of variables of d which are handled like regressors.
            Each key representing the name of a variable, values are list with the names of the nodes for
            which this variable must be retrieved
    :param pars: dict of parameters: 'h_b', 'h_f' integers, backward and future embeddings
                                     'x_reduction', 'y_reduction', 'f_reduction', dict with the following keys:
                                    {'type','values'}. Type must be in ['bins','aggregated_selection','min','max',
                                    'mean'] while 'values' is a list of bin lenghts if type=='bins', or an array in
                                    accumarray style if type=='aggregated_selection'.
                                     'week_day', 'hour', 'vacation': boolean, if true these features are retrieved from
                                     the time index of the data frames and added to regressors.

    :param f: pandas.Dataframe of forecasted variables, which are not divided into nodes. It is assumed they are
            encoded in lists inside the pd.Dataframe. They extend from t=0 up to the prediction horizon
    :param g: pandas.Dataframe of global variables, which are not divided into nodes

    :param forecasts_names: list variables of f to handle like forecasted regressors.
    :param global_names: list variables of f to handle like global regressors.
    :param h: embedding dimension (backward and forward steps)
    :param reduction_bins: if not None, the dataset is reduced according to the bins
    :return: numpy ndarray
    """

    # verify that d and forecasts are aligned
    var_keys = list(d.keys())
    sign_keys = list(d[var_keys[0]].keys())
    t_d = d[var_keys[0]][sign_keys[0]].index

    # --------------------------- retrieve past regressors from node variables in d ---------------------------
    x_pd = pd.DataFrame({})
    for v in var_names.keys():
        node_names = var_names[v]

        if not type(node_names) in [list,tuple]:
            raise ValueError('values of var_names dict must be tuples or list of node names')
        if v not in d.keys():
            raise ValueError('key %s in var_names dict is not in d dict' % v)

        var = d[v]
        for node in node_names:
            x_pd[v+'_'+node] = var[node]

    # --------------------------- retrieve past regressors from global variables in g ---------------------------
    if g is not None:
        for gn in global_names:
            if gn not in g.columns():
                raise ValueError('key %s in global_names dict is not in d dict' % gn)
            var = d[gn]
            x_pd[gn] = var

    # --------------------------- retrieve targets from node variables in d ---------------------------
    y_pd = pd.DataFrame({})
    for v in target_names.keys():
        node_names = target_names[v]

        if not type(node_names) in [list, tuple]:
            raise ValueError('values of var_names dict must be tuples or list of node names')
        if v not in d.keys():
            raise ValueError('key %s in var_names dict is not in d dict' % v)

        var = d[v]
        for node in node_names:
            y_pd['target'+'_'+node] = var[node]

    # --------------------------- hankelize, reduce, shift ---------------------------
    x_pd, y_pd = format_pandas_df(x_pd, pars, y_pd)

    # --------------------------- mark past regressors with '_past' signature ---------------------------
    x_keys = list(x_pd.keys())
    x_keys = [a + '_past' for a in x_keys]
    x_pd.columns = x_keys

    # --------------------------- add time features if requested ---------------------------
    x_pd = add_time_features(x_pd,pars)

    # --------------------------- get only current time values ---------------------------
    x_0 = pd.DataFrame(index=x_pd.index)
    for k in x_pd.keys():
        x_v = np.vstack(x_pd[k].values)
        x_0[k] = np.atleast_2d(x_v)[:,[-1]]

    # --------------------------- convert to matrices ---------------------------
    x, t_dummy = ut.list_df_to_matrix(x_pd)

    # --------------------------- retrive forecasted variables from f ---------------------------
    if f is not None:
        t_f = f.index
        if not t_d.equals(t_f):
            raise ValueError('pandas dataframe in d and forecasts dataframe do not have the same time index')
        f_pd = f[forecasts_names]
        f_pd = f_pd.iloc[pars['h_b']:-pars['h_f'] + 1, :]
        if pars['f_reduction']:
            f_pd = ut.reduce_dataset(f_pd, pars['f_reduction'])
        # mark future regressors with '_future' signature
        f_keys = list(f_pd.keys())
        f_keys = [a + '_future' for a in f_keys]
        f_pd.columns = f_keys
        f_mat, t_dummy = ut.list_df_to_matrix(f_pd)
        # concatenate X and F matrices
        X = np.hstack([x, f_mat])
        # concat forecasts with past data
        f_pd.index = x_pd.index
        x_pd = pd.concat([x_pd, f_pd], axis=1)

    y, t_dummy = ut.list_df_to_matrix(y_pd)
    y_hat_persistence, y_0 = get_y_persistence(y_pd,pars)
    t = y_pd.index

    return x_pd,y_pd,x,y,t,x_0,y_0,y_hat_persistence


def format_pandas_df(x,pars,y=None, keep_cols=None):

    # --------------------------- get Hankel data frames (data frames of lists) ---------------------------
    t_index = x.index
    x_h = pd.DataFrame(index=t_index[pars['h_b']-1:])
    for k in x.keys():
        hankel = ut.hankelize_matrix(x[k].values.reshape(-1,1), pars['h_b'], keep_cols)
        x_h[k] = hankel.tolist()
    # --------------------------- reduce data frame if requested ---------------------------
    if pars['x_reduction']:
        x_h = ut.reduce_dataset(x_h, pars['x_reduction'])

    if y is None:
        return x_h
    else:
        y_h = pd.DataFrame(index=t_index[:len(t_index) - pars['h_f'] + 1])
        for k in y.keys():
            y_h[k] = ut.hankelize_matrix(y[k].values.reshape(-1,1), pars['h_f']).tolist()
        if pars['y_reduction']:
            y_h = ut.reduce_dataset(y_h, pars['y_reduction'])

        # --------------------------- shift matrices according to embedding dimension ---------------------------
        x_h = x_h.iloc[0:-pars['h_f'], :]
        y_h = y_h.iloc[pars['h_b']:, :]


        return x_h, y_h


def get_y_persistence(y:pd.DataFrame,pars:dict):
    """
    Get persistence predictions, given a pd.DataFrame of lists
    :param y: pd.DataFrame of lists
    :param pars: dict
    :return:
    """
    y_0 = pd.DataFrame(index=y.index)
    for k in y.keys():
        y_v = np.vstack(y[k].values)
        y_0[k] = np.atleast_2d(y_v)[:,[0]]

    # get persistence forecasts
    y_hat_per = ut.hankelize_matrix(np.copy(np.vstack([y_0.values[0:pars['h_f']], y_0.values]).reshape(-1,1)),pars['h_f'])
    y_hat_per = ut.matrix_to_list_df(y_hat_per[:-1,:],y_0.index)
    return y_hat_per, y_0


def add_time_features(x:pd.DataFrame, pars:dict):
    """
    Add time features to the pandas dataframe, based on the specs in pars
    :param x: a pd.DataFrame
    :param pars: dict, with boolean entries {'week_day','hour','vacation'}
    :return:
    """
    if pars['week_day']:
        x['week_day'] = x.index.weekday
    if pars['hour']:
        x['hour'] = x.index.hour
    if pars['vacation']:
        vacations = pd.read_csv('data/vacations.csv')
        is_vacation = []
        for i in np.arange(len(vacations)):
            is_vacation_day = x.index.date == pd.to_datetime(vacations.values.ravel()).date[i]
            is_vacation.append(np.asanyarray(is_vacation_day, int).reshape(-1, 1))
        is_vacation = np.sum(np.hstack(is_vacation), axis=1)
        x['vacation'] = is_vacation

    return x


def get_past_mask(bins: Union[list, np.ndarray], n_sa: int, periods: Union[list,np.ndarray] = None, include_last: bool = True) -> np.ndarray:
    '''
    Get a selection mask for past inputs. If periods is not specified, the mask is full
    :param bins: aggregation bins of past regressors, expressed in an accumarray notation
    :param n_sa: number of steps to be predicted
    :param periods: list of seasonalities to be taken into account
    :return: a matrix of size (n_sa,n_agg), where n_agg is the number of unique elements of bins
    '''

    n_agg = len(np.unique(bins))
    if periods is not None:
        mask = np.zeros((n_sa, n_agg))
        mask_index = []
        for s in range(n_sa):
            mask_s = np.zeros((1,n_agg))
            mask_index_s = []
            if include_last:
                mask_s[0,-1] = 1 # always keep last observation in the regressors' set
                mask_index_s.append(n_agg-1)
            if np.all([p_i <= n_agg for p_i in periods]):
                for p in periods:
                    mask_s[0,-p+s] = 1
                    mask_index_s.append(n_agg-p+s)
            mask_index.append(np.hstack(mask_index_s).reshape(1,-1))
            mask[s,:] = mask_s
        mask_index = np.vstack(mask_index)
    else:
        mask = np.ones((n_sa, n_agg))
        mask_index = np.ones((n_sa, n_agg))*np.arange(0, n_agg)

    return np.asanyarray(mask,dtype=bool), mask_index


def get_future_mask(bins: Union[list, np.ndarray], n_sa: int, n_future_reg, mode: str):
    '''
    Get a selection mask for past inputs. If periods is not specified, the mask is full
    :param bins: aggregation bins of past regressors, expressed in an accumarray notation
    :param n_sa: number of steps ahead
    :param n_future_reg: number of future regressors
    :param periods: list of seasonalities to be taken into account
    :return: a matrix of size (n_sa,n_agg), where n_agg is the number of unique elements of bins
    '''

    n_agg = len(np.unique(bins))     # number of aggregated future regressors
    mask = np.zeros((n_sa, n_future_reg))

    for s in range(n_sa):
        if mode == 'up_to':
            mask[s, :s+1] = 1
        elif mode == 'full':
            mask[s,s] = 1
        elif mode == 'hourly_up_to':
            k = np.asanyarray(np.floor(s / (n_agg / n_future_reg)), int) + 1
            mask[s, 0:k] = 1
        elif mode == 'hourly':
            k = np.asanyarray(np.floor(s / (n_agg / n_future_reg)),int)
            mask[s, k] = 1
    mask_ix = [np.where(future_mask_i)[0] for future_mask_i in mask]
    return np.asanyarray(mask, dtype=bool), mask_ix