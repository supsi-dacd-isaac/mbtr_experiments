import numpy as np
import pandas as pd
from utils.qsturng import qsturng
from scipy.stats import friedmanchisquare, rankdata
import matplotlib.pyplot as plt
from analysis.plot_utils import set_figure


def nemenyi(data, conf_level=0.9, sort=True, labels=None, k_cv=1):
    assert len(data.shape)!=1, 'must have a 2D array, while dims are {}'.format(data.shape)
    n, m = data.shape
    n *= k_cv
    # run a preliminary pairwise Friedman test
    fr_stat, fr_p_val = friedmanchisquare(*data)
    are_identical = True if fr_p_val > 1 - conf_level else False

    # Nemenyi critical distance and bounds of intervals
    r_stat = qsturng(conf_level, m, np.inf) * ((m * (n+1)) / (12 * n)) ** 0.5

    # Rank methods for each time series
    ranks_matrix = np.vstack(list(map(rankdata, data)))
    ranks_means = np.mean(ranks_matrix, axis=0)
    ranks_intervals = np.vstack([ranks_means-r_stat, ranks_means + r_stat])

    return ranks_means, ranks_intervals, fr_p_val, fr_stat, are_identical


def nemenyi_unrolled_plot(data, ylabel, xlabel, k_cv=1, rot=0, **set_fig_kwargs):
    nem = pd.DataFrame(columns=['means', 'l_b', 'u_b'])
    models = data.columns
    for i in np.unique(data.index):
        d_i = data.loc[data.index == i]
        ranks_means, ranks_intervals, fr_p_val, fr_stat, are_identical = nemenyi(d_i.values, k_cv=k_cv)
        print('index {},  are identical: {}'.format(i, are_identical))
        nem_i_dat = np.hstack([ranks_means.reshape(-1,1),ranks_intervals.T])
        nem_i = pd.DataFrame(nem_i_dat, columns=['means', 'l_b', 'u_b'], index=np.tile(i,len(models)))
        nem_i = pd.concat([nem_i, pd.Series(models, name='model', index=nem_i.index)], axis=1)
        nem = pd.concat([nem, nem_i], axis=0)

    fig, ax = set_figure(**set_fig_kwargs)
    cm = plt.get_cmap('plasma', len(models)*2)
    for i, ind in enumerate(np.unique(data.index)):
        nem_i = nem.loc[nem.index==ind]
        for j,m in enumerate(models):
            nem_i_m = nem_i.loc[nem_i['model']==m]
            x = i + j/(len(models) * 1.2) -0.5
            w = 0.8/len(models)
            plt.bar(x, nem_i_m['u_b']-nem_i_m['l_b'], bottom=nem_i_m['l_b'], color=cm(j), width=w, alpha=0.3)
            plt.scatter(x, nem_i_m['means'], marker='d', color=cm(j), s=w*50/len(np.unique(data.index)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(np.unique(data.index))))
    ax.set_xticklabels(np.unique(data.index), rotation=rot)
    y_min = nem['l_b'].min() * (1 - np.sign(nem['l_b'].min()) * 0.1) - 0.2
    y_max = np.max(nem['u_b'].max()) * (1 + np.sign(nem['u_b'].max()) * 0.1)
    ax.set_ylim(y_min, y_max)
    plt.legend(models, loc='upper right', fontsize='x-small')
    return fig, ax