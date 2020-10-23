import pandas as pd
import numpy as np
from utils.cross_val import get_cv_results
import matplotlib.pyplot as plt
from utils.nemenyi import nemenyi, nemenyi_unrolled_plot

plot_pars = {"size": (5, 2.5),
             "font_scale": 0.7,
             "w": 0.3,
             "h": 0.2,
             "b": 0.2}

# ----------------------------------------------------------------------------------
# -------------------------------- Fourier Smoothing -------------------------------
# ----------------------------------------------------------------------------------

# variable: MAPE, independent var: n harmonics, groups: time series

results = np.load('data/cv_res_fourier_144.npy', allow_pickle=True).item()
k_cv = len(results[list(results.keys())[0]])

resall = pd.DataFrame()
for k in results.keys():
    results_k = get_cv_results(results[k])
    results_k['series'] = k
    resall = pd.concat([resall, pd.DataFrame(results_k)], axis=0)

# rename columns for plot
resall = resall[['mape_miso', 'mape_mimo', 'mape']]
resall.columns = ['miso', 'mimo', 'mbt']
fig, ax = nemenyi_unrolled_plot(resall, 'rank [-]', 'n freq [-]', k_cv=k_cv, rot=60, **plot_pars)
plt.title('MAPE')
plt.savefig('figs/stats_fourier.pdf')
plt.show()

# ----------------------------------------------------------------------------------
# --------------------------------- Hierarchical -----------------------------------
# ----------------------------------------------------------------------------------

# variable: MAPE, independent var: group level, groups: first 3 steps ahead

results = np.load('data/hierarchical_scores.npy', allow_pickle=True)
k_cv = len(results)
results = get_cv_results(results)

sa_filter = np.arange(3).astype(float)
group_filter = ['bu', 'rec', 'mgb']
score_filter = 'mape'
res = pd.DataFrame(columns=group_filter)
for k, v in results.items():
    if k not in group_filter:
        continue
    v = v[v['sa'].isin(sa_filter)]
    v = v[score_filter]
    res[k] = v

# rename columns for plot
res.columns = ['bu' ,'rec', 'mbt']
fig, ax = nemenyi_unrolled_plot(res, 'rank [-]', 'aggregation group [-]', k_cv=k_cv, **plot_pars)
plt.title('MAPE')
plt.savefig('figs/stats_hierarchical.pdf')
plt.show()


# ----------------------------------------------------------------------------------
# --------------------------------- Quantiles --------------------------------------
# ----------------------------------------------------------------------------------

# variable: QS, independent var: quantile, groups: 24 steps ahead

results_all = np.load('data/cv_res_quantiles_all_squared_partial.npy', allow_pickle=True).item()
refit = get_cv_results(results_all['refit']['all'])
squared_refit = get_cv_results(results_all['suqared_refit']['all'])
results_all = get_cv_results(results_all['no_refit']['all'])

results_all['mbt'] = results_all['mgb']
del results_all['mgb']

results_all['mbt refit'] = refit['mgb']
results_all['mbt lin-quad refit'] = squared_refit['mgb']


n_quantiles = 11
alphas = np.linspace(1/n_quantiles, 1-1/n_quantiles, n_quantiles)

group_filter = ['mimo', 'mbt refit', 'mbt lin-quad refit']
score_filter = 'skill'
res = pd.DataFrame()
for k in group_filter:
    res_k = pd.DataFrame(results_all[k][score_filter], columns=np.arange(24), index=alphas)
    res_k.index.name = 'quantile'
    res_k.index = np.round(res_k.index, 2)
    res_k = res_k.reset_index().melt('quantile').set_index('quantile')[['value']]
    res_k.columns = [k]
    res = pd.concat([res, res_k], axis=1)

fig, ax = nemenyi_unrolled_plot(res, 'rank [-]', r'$\alpha$ [-]', k_cv=k_cv, **plot_pars)
plt.title('QS')
plt.savefig('figs/stats_quantile_QS.pdf')
plt.show()

# variable: reliability distance, independent var: quantile, groups: 24 steps ahead

group_filter = ['mimo', 'mbt refit', 'mbt lin-quad refit']
score_filter = 'reliability'
res = pd.DataFrame()
for k in group_filter:
    res_k = pd.DataFrame(results_all[k][score_filter], columns=np.arange(24), index=alphas)
    # retrieve reliability distance from perfect reliability
    res_k = (res_k - alphas.reshape(-1, 1)).abs()
    res_k.index.name = 'quantile'
    res_k.index = np.round(res_k.index, 2)
    res_k = res_k.reset_index().melt('quantile').set_index('quantile')[['value']]
    res_k.columns = [k]
    res = pd.concat([res, res_k], axis=1)

fig, ax = nemenyi_unrolled_plot(res, 'rank [-]', r'$\alpha$ [-]', k_cv=k_cv, **plot_pars)
plt.title('reliability')
plt.savefig('figs/stats_quantile_reliability.pdf')
plt.show()

