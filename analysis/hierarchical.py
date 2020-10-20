import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from utils.cross_val import get_cv_results
from analysis.plot_utils import set_figure

size = (5,4)
font_scale = 0.7
results = np.load('data/hierarchical_scores.npy', allow_pickle=True)
results = get_cv_results(results)
results['mbt'] = results['mgb']
name = 'reconciliation_RMSE'
f,ax = set_figure(size=size,subplots=(2,1),w=0.3, h=0.2,b=0.2,font_scale=font_scale)

results['bu']['type'] = results['bu'].index
results['rec']['type'] = results['rec'].index
results['mbt']['type'] = results['mgb'].index
results['bu']['method'] = 'bu'
results['rec']['method'] = 'rec'
results['mbt']['method'] = 'mbt'
res_k = pd.concat([results['bu'],results['rec'],results['mbt']])

sb.lineplot(x='sa', y='mape', data=res_k, hue='type', style='method',ax=ax[1])
ax[1].set_ylabel('MAPE [-]')
plt.legend(fontsize='small', ncol=2,handleheight=0.4, labelspacing=0.05, loc='lower right')

plt.sca(ax[0])
sb.lineplot(x='sa', y='rmse', data=res_k, hue='type', style='method',ax=ax[0])
plt.legend(fontsize='small', ncol=2,handleheight=0.4, labelspacing=0.05, loc='lower right')
ax[0].set_ylabel('RMSE [kW]')

ax[0].set_xlabel('step ahead [h]')
ax[1].set_xlabel('step ahead [h]')

plt.savefig('figs/{}.pdf'.format(name))
