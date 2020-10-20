import numpy as np
from mbtr.mbtr import MBT
import matplotlib.pyplot as plt
from utils.cross_val import cv, noise_vsc, get_scores, parallel_cv
from sklearn.linear_model import RidgeCV, LinearRegression
import pickle as pk
import wget

try:
    with open('data/vsc_data.pk', 'rb') as f:
        data = pk.load(f)
except:
    wget.download('https://zenodo.org/record/4108561/files/vsc_data.pk?download=1', 'data/vsc_data.pk')
    with open('data/vsc_data.pk', 'rb') as f:
        data = pk.load(f)

x = data['x']
y = data['y']
h_day = np.array(data['trafo'].index.hour).reshape(-1,1)
week_day = np.array(data['trafo'].index.weekday).reshape(-1,1)
x_trafo = np.hstack([data['trafo'].values.reshape(-1,1),h_day, week_day])

n_boosts = 100
min_n_leaf = data['x'].shape[1]*10

mbt_pars = {'n_boosts': n_boosts,
            'n_q': 10,
            'early_stopping_rounds': 3,
            'min_leaf':data['x'].shape[1]*10,
            'loss_type': 'mse',
            'lambda_leaves': 0.1,
            'lambda_weights': 0.01
            }
mbt_pars_lin = mbt_pars.copy()
mbt_pars_lin['loss_type'] = 'linear_regression'
mbt_pars_lin['min_leaf'] = data['x'].shape[1]*30


def cv_fun(x_tr,y_tr,x_te, y_te, x_lingb_tr, x_lingb_te):
    models = []
    scores = {}
    x_tr_d, y_tr_d = np.diff(x_tr, axis=0), np.diff(y_tr, axis=0)
    x_te_d, y_te_d = np.diff(x_te, axis=0), np.diff(y_te, axis=0)
    x_lingb_tr = x_lingb_tr[0:-1,:]
    x_lingb_te = x_lingb_te[0:-1, :]

    # fit VSC canonical
    models.append(LinearRegression())
    models.append(RidgeCV(alphas=10 ** np.linspace(-2, 8, 20)))
    models.append(MBT(**mbt_pars_lin))
    models.append(MBT(**mbt_pars))

    names = ['linear','ridge','mbt lin','mbt']
    y_mean = np.ones_like(y_te_d) * np.mean(y_te_d, axis=0)
    for m, model in enumerate(models):
        print('start fitting model {}'.format(model))
        if names[m] == 'mbt lin':
            scores[names[m]] = noise_vsc(x_lingb_tr,y_tr_d,x_tr_d, x_lingb_te, x_te_d,y_te_d,y_mean, model)
        else:
            model_fitted = model.fit(x_tr_d, y_tr_d)
            y_hat = model_fitted.predict(x_te_d)
            scores[names[m]] = get_scores(y_hat, y_te_d, y_mean)
            print(scores[names[m]])
    plt.close('all')
    return scores


# ----------------------------------------------------
# perform CV using data of the trafo to build the tree
# ----------------------------------------------------
k = 10
cv_results = cv(x, y, k, cv_fun, x_build=x_trafo)
np.save('data/vsc_results.npy', cv_results)
print(cv_results)

# ----------------------------------------------------
# perform CV using only meteo data to build the tree
# ----------------------------------------------------
h_day = np.array(data['trafo'].index.hour).reshape(-1,1)
week_day = np.array(data['trafo'].index.weekday).reshape(-1,1)
x_meteo = np.hstack([data['meteo'],h_day, week_day])
cv_results = cv(x, y, k, cv_fun, x_build=x_meteo)
np.save('data/vsc_results_meteo.npy', cv_results)
print(cv_results)