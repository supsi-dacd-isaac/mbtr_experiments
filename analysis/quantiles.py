import numpy as np
from utils.cross_val import get_cv_results
from analysis.plot_utils import plot_reliability, plot_reliability_tilted, plot_QS, plot_crsp, plot_reliability_diff, plot_QS_diff
font_scale = 0.7
size = (4.5,3)


# ----------------------------------- compare normal, refitted and squared-refitted -------------------------

results_all = np.load('data/cv_res_quantiles_all_squared_partial.npy', allow_pickle=True).item()
refit = get_cv_results(results_all['refit']['all'])
squared_refit = get_cv_results(results_all['suqared_refit']['all'])
results_all = get_cv_results(results_all['no_refit']['all'])

results_all['mbt'] = results_all['mgb']
del results_all['mgb']

results_all['mbt refit'] = refit['mgb']
results_all['mbt lin-quad refit'] = squared_refit['mgb']

plot_QS_diff(results_all,size,font_scale=font_scale)
plot_reliability_diff(results_all,size,font_scale=font_scale)
plot_crsp(results_all,size,font_scale=font_scale)
plot_QS(results_all,size,font_scale=font_scale)
plot_reliability(results_all,size,font_scale=font_scale)
plot_reliability_tilted(results_all,size,font_scale=font_scale)

