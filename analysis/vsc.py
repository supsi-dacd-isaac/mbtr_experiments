import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from utils.cross_val import get_cv_results
from analysis.plot_utils import set_figure
from collections import OrderedDict
from matplotlib.ticker import FormatStrFormatter

size = (4.5,3)
font_scale = 0.7
ns_control = [1]
results = np.load('data/vsc_results.npy', allow_pickle=True)
results = get_cv_results(results)

results_meteo = np.load('data/vsc_results_meteo.npy', allow_pickle=True)
results_meteo = get_cv_results(results_meteo)

# pretty rename
results['mbt lin meteo'] = results_meteo['mbt lin']
results['mbt meteo'] = results_meteo['mbt']


summary = OrderedDict({'rmse':{},'rmse_norm':{}})
for k,v in results.items():
    if k in ['mbt', 'mbt meteo'] :
        continue
    summary['rmse'][k] = v['rmse']
    summary['rmse_norm'][k] = v['norm_rmse']


f,ax = set_figure(size=size,w=0.3, h=0,b=0.2,font_scale=font_scale)
summary = pd.DataFrame(summary)
summary.reset_index(level=0, inplace=True)
summary.rename(columns={'index':'model'}, inplace=True)
summary = summary.melt('model',var_name='metric',value_name='RMSE [V]')

sb.set_color_codes("pastel")
sb.barplot(x='model', y='RMSE [V]', hue='metric', data=summary, linewidth=0.2)

plt.savefig('figs/vsc_rmse.pdf')
plt.show()

# noise analysis
results = np.load('data/vsc_results_meteo.npy', allow_pickle=True)
cm = plt.get_cmap('plasma',10)

f,ax = set_figure(size=size,w=0.3, h=0,b=0.2,font_scale=font_scale)

rmse_norm = pd.DataFrame({})
# retrieve the mean mape of the input:
x_mape = []
for f,r in enumerate(results):
    x_mape.append([r['mbt lin']['x_scores'][k]['mape'] for k in r['mbt lin']['x_scores'].keys()])
x_mape = np.vstack(x_mape).T
x_mape_mean = np.mean(x_mape,axis=1)
for f,r in enumerate(results):
    rmse_norm[f] = pd.DataFrame(r['mbt lin']['scores_noise_partial_cv']).T['norm_rmse']

rmse_norm.index = ['{:0.2}'.format(k) for k in x_mape_mean]
rmse_norm.index = rmse_norm.index.set_names(['input MAPE [-]'])
sb.set_context(rc = {'patch.linewidth': 0.0})
boxprops = dict(linewidth=0, color=cm(4),alpha=0.3)
ax =sb.boxplot(data =rmse_norm.reset_index().melt(id_vars='input MAPE [-]',value_name='norm RMSE [-]'), x='input MAPE [-]',y='norm RMSE [-]',
               color=cm(4),boxprops=boxprops)
ax.set_ylabel('normalized RMSE [-]')

plt.hlines(summary['RMSE [V]'][5],-0.5,9.5,linestyles='--',color=cm(6),label='ridge norm RMSE')
plt.legend(loc=(0.45,0.8))

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.savefig('figs/vsc_noise.pdf')
plt.show()