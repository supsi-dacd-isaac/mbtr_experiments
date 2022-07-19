import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from utils.cross_val import get_cv_results
from analysis.plot_utils import set_figure

font_scale = 0.7
size = (4.5,4)

results = np.load('data/cv_res_m4.npy', allow_pickle=True).item()
res_miso, res_mimo, rmse_miso,rmse_mimo,mape_mimo,mape_miso = [{},{},{},{},{},{}]
res = []
for k in results.keys():
    results_k = get_cv_results(results[k])
    res.append(pd.DataFrame(results_k, index=[k]))
res = pd.concat(res)
res['mape wrt mimo'] = res['mape']/res['mape_mimo']
res['mape wrt miso'] = res['mape']/res['mape_miso']
res['rmse wrt mimo'] = res['rmse']/res['rmse_mimo']

res['rmse wrt miso'] = res['rmse']/res['rmse_miso']

fig,ax = set_figure(size, (2, 1),font_scale=font_scale,w=0,h=0, sharex=True)
sb.histplot(data=res.loc[:,['mape wrt mimo', 'mape wrt miso']], kde=True, bins=50, ax=ax[0])
sb.histplot(data=res.loc[:,['rmse wrt mimo', 'rmse wrt miso']], kde=True, bins=50, ax=ax[1])
ax[0].vlines(1, *ax[0].get_ylim(), linewidth=1.5, color='k', linestyle='--')
ax[1].vlines(1, *ax[1].get_ylim(), linewidth=1.5, color='k', linestyle='--')
plt.savefig('figs/m4_regression.pdf')
plt.show()


results = np.load('data/cv_res_m4_fourier.npy', allow_pickle=True).item()

res_miso, res_mimo, rmse_miso,rmse_mimo,mape_mimo,mape_miso = [{},{},{},{},{},{}]
for k in results.keys():
    results_k = get_cv_results(results[k])
    mape = pd.DataFrame(results_k['mape'],index=['mape']).T/results_k['mape_miso']
    rmse = pd.DataFrame(results_k['rmse'], index=['rmse']).T/ results_k['rmse_miso']
    res_miso[k] = pd.concat([mape,rmse],axis=1)

    mape = pd.DataFrame(results_k['mape'],index=['mape']).T / results_k['mape_mimo']
    rmse = pd.DataFrame(results_k['rmse'], index=['rmse']).T / results_k['rmse_mimo']

    res_mimo[k] = pd.concat([mape,rmse],axis=1)

    rmse_miso[k] = results_k['rmse_miso']
    mape_miso[k] = results_k['mape_miso']
    rmse_mimo[k] = results_k['rmse_mimo']
    mape_mimo[k] = results_k['mape_mimo']


#fig, ax = set_figure(size, (2,2),font_scale=font_scale,w=0,h=0)

fig, ax = set_figure((5, 4), (2,2),font_scale=font_scale, w=0.25, h=0.25, l=0.1, r=0.8)

# get terminal stages
pdfs = pd.DataFrame({})
pdfs['mape_mimo'] = [v['mape'].iloc[-1] for v in res_mimo.values()]
pdfs['mape_miso'] = [v['mape'].iloc[-1] for v in res_miso.values()]
pdfs['rmse_mimo'] = [v['rmse'].iloc[-1] for v in res_mimo.values()]
pdfs['rmse_miso'] = [v['rmse'].iloc[-1] for v in res_miso.values()]

# set distribution bars
distplot_width = 0.05
dax = []
sb.set_style('whitegrid')
for i, a in enumerate(ax.ravel()):
    x0, y0, b_w, b_h = a.get_position().bounds
    dax.append(plt.axes([x0+b_w+0.01, y0, distplot_width, b_h]))
    dax[-1].set_xticks([])
    dax[-1].set_yticks([])
    sb.histplot(data=pdfs, y=pdfs.columns[i])
    sb.despine(ax=dax[-1])
    dax[-1].hlines(1, *dax[-1].get_xlim()*np.array([0, 1.3]), linestyles='--', colors='gray')

lw = 2
s = 10
alpha = 0.3
t = np.asanyarray(list([res_mimo.values()][0])[0]['mape'].keys())

cm = plt.get_cmap('plasma',256)
for k,v in res_mimo.items():
    color = np.minimum(1,mape_mimo[k]/np.quantile(list(mape_mimo.values()),0.9))
    ax[0, 0].plot(t,v['mape'], color=cm(color),alpha=alpha)
    ax[0, 0].scatter(v.index[np.argmin(v['mape'])],np.min(v['mape']),color=cm(color),marker='o',s=s)

ax[0, 0].plot(t,np.ones_like(t), color='grey', linestyle='--', linewidth=lw)
ax[0, 0].set_ylabel('MAPE ratio [-]')
ax[0, 0].set_title('mimo')
ax[0, 0].set_xticklabels([])

for k,v in res_miso.items():
    color = np.minimum(mape_mimo[k]/np.quantile(list(mape_mimo.values()), 0.9),1)
    ax[0, 1].plot(t,v['mape'], color=cm(color),alpha=alpha)
    ax[0, 1].scatter(v.index[np.argmin(v['mape'])],np.min(v['mape']),color=cm(color),marker='o', s=s)

ax[0, 1].plot(t, np.ones_like(t), color='grey', linestyle='--', linewidth=lw)
ax[0, 1].set_title('miso')
ax[0, 1].set_yticklabels([])
ax[0, 1].set_xticklabels([])

y_lims = (np.minimum(ax[0, 0].get_ylim()[0],ax[0, 1].get_ylim()[0]),np.maximum(ax[0, 0].get_ylim()[1],ax[0, 1].get_ylim()[1]))
ax[0, 0].set_ylim(y_lims)
ax[0, 1].set_ylim(y_lims)

for k, v in res_mimo.items():
    color = np.minimum(mape_miso[k]/np.quantile(list(mape_mimo.values()), 0.9),1)
    ax[1, 0].plot(t, v['rmse'], color=cm(color),alpha=alpha)
    ax[1, 0].scatter(v.index[np.argmin(v['rmse'])],np.min(v['rmse']),color=cm(color),marker='o',s=s)

ax[1,0].plot(t,np.ones_like(t), color='grey', linestyle='--', linewidth=lw)

ax[1,0].set_ylabel('RMSE ratio [-]')

for k, v in res_miso.items():
    color = np.minimum(1,mape_miso[k]/np.quantile(list(mape_mimo.values()), 0.9))
    ax[1, 1].plot(t, v['rmse'], color=cm(color),alpha=alpha)
    ax[1, 1].scatter(v.index[np.argmin(v['rmse'])],np.min(v['rmse']),color=cm(color),marker='o',s=s)

ax[1, 1].set_yticklabels([])
ax[1, 1].plot(t,np.ones_like(t), color='grey', linestyle='--', linewidth=lw)

y_lims = (np.minimum(ax[1, 0].get_ylim()[0],ax[1, 1].get_ylim()[0]),np.maximum(ax[1, 0].get_ylim()[1],ax[1, 1].get_ylim()[1]))
ax[1, 0].set_ylim(y_lims)
ax[1, 1].set_ylim(y_lims)
ax[1, 0].set_xlabel('n freq [-]')
ax[1, 1].set_xlabel('n freq [-]')

x0, y0, b_w, b_h = dax[-1].get_position().bounds
#plt.subplots_adjust(bottom=0.1, right=x0+0.1, top=0.9)
cax = plt.axes([x0+0.08, 0.1, 0.025, 0.8])
sm = plt.cm.ScalarMappable(cmap=cm)
cb =plt.colorbar(mappable=sm, cax=cax)
cb.set_label('MAPE [-]')
cb.ax.set_yticklabels(['{:0.2}'.format(i) for i in  np.linspace(np.min(list(mape_mimo.values())),np.quantile(list(mape_mimo.values()), 0.9),6)],rotation=90)

# adjust dist axes limits
for d, a in zip(dax, ax.ravel()):
    d.set_ylim(a.get_ylim())

plt.savefig('figs/m4_fourier_smoothing.pdf')