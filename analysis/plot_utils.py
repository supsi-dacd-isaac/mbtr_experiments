import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


def set_figure(size, subplots=(1,1), context='paper', style='darkgrid',font_scale = 1, l=0.2, w=0.1, h=0.1, b=0.1):
    sb.set(context=context,style=style, font_scale=font_scale)
    fig, ax = plt.subplots(subplots[0],subplots[1], figsize=size)
    plt.subplots_adjust(left=l, wspace=w, hspace=h, bottom=b)
    return fig, ax


def plot_logistic_loss(alpha, font_scale):
    N = 200
    size = (4,3)
    x = (np.arange(N)-N/2)/10
    shift = np.log((1 - alpha) / alpha)
    softplus = np.log(np.exp(x)+np.exp(-shift)) + (alpha-1)*x
    loss_grad_i = (np.exp(x + shift) / (1 + np.exp(x + shift)) + alpha - 1)
    hessian_i = np.exp(x + shift) / ((1 + np.exp(x + shift)) ** 2)

    shift = 0 #- np.log((1 - alpha) / alpha)
    softplus_s = np.log(np.exp(x)+np.exp(-shift)) + (alpha-1)*x
    loss_grad_i_s = (np.exp(x + shift) / (1 + np.exp(x + shift)) + alpha - 1)
    hessian_i_s = np.exp(x + shift) / ((1 + np.exp(x + shift)) ** 2)

    colors = plt.get_cmap('plasma',12)

    sb.set(context='paper',style='dark', font_scale=font_scale)
    plt.figure(figsize=size, tight_layout=True)
    ax1 = plt.subplot(111)
    l1 = plt.plot(x,softplus, c=colors(2))
    l1_s = plt.plot(x, softplus_s, c=colors(2), linestyle='--')

    ax2 = ax1.twinx()
    l2 = plt.plot(x, loss_grad_i, c=colors(5), label=r'$\tilde{\mathcal{l}}^{\ \prime}_q$')
    l3 = plt.plot(x, hessian_i, c=colors(7), label=r'$\tilde{\mathcal{l}}^{\ \prime\prime}_q$')

    l2_s = plt.plot(x, loss_grad_i_s, c=colors(5), linestyle='--', label=r'$\tilde{\mathcal{l}}^{\ \prime}_q$')
    l3_s = plt.plot(x, hessian_i_s, c=colors(7), linestyle='--', label=r'$\tilde{\mathcal{l}}^{\ \prime\prime}_q$')

    ax1.legend(l1+l2+l3, [r'$\tilde{\mathcal{l}}_q$',r'$\tilde{\mathcal{l}}^{\ \prime}_q$',r'$\tilde{\mathcal{l}}^{\ \prime\prime}_q$'], loc=0, fontsize='small')
    ax1.set_ylabel(r'$\tilde{\mathcal{l}}_q$')
    ax2.set_ylabel(r"$\tilde{\mathcal{l}}^{\ \prime}_q, \tilde{\mathcal{l}}^{\ \prime\prime}_q$")
    ax1.vlines(0,ax1.get_ylim()[0],ax1.get_ylim()[1]+0.1,linestyles='--',color='gray')
    ax2.hlines(0,ax2.get_xlim()[0],ax2.get_xlim()[1],linestyles='--',color='gray')
    ax1.set_xlabel(r'$\epsilon_{\tau}$')

    plt.savefig('figs/quantile_loss.pdf')


def plot_crsp(results,size,font_scale):
    name = 'mean_crps'
    f,ax = set_figure(size=size,subplots=(2,1),w=0.3, h=0,b=0.2, font_scale=font_scale)
    for k,v in results.items():
        plt.sca(ax[0])
        plt.plot(v['crps_t'], label=k)
    plt.legend(fontsize='small')
    plt.ylabel(r'$Qs \quad [kW]$')

    for k,v in results.items():
        plt.sca(ax[1])
        plt.plot(v['crosses'], label=k)
    plt.legend(fontsize='small')
    plt.ylabel(r'$\overline{\chi} \quad [-]$')
    plt.xlabel('step ahead [h]')
    plt.savefig('figs/{}.pdf'.format(name))


def plot_reliability(results,size,font_scale):
    name = 'reliability'
    n_res = len(results)
    f,ax = set_figure(size=size,subplots=(1,n_res),w=0.05, h=0,b=0.2,font_scale=font_scale)
    z = 0
    n_q = results[list(results.keys())[0]]['reliability'].shape[0]
    alphas = np.linspace(1 / n_q, 1 - 1 / n_q, n_q)
    for k,v in results.items():
        plt.sca(ax[z])
        n = v['reliability'].shape[1]
        cm = plt.get_cmap('viridis',n)
        plt.plot(alphas,alphas)
        for i in range(n):
            d = v['reliability'][:, i]
            plt.plot(alphas, d, label=k, c = cm(i), alpha=0.8)
        plt.xticks(rotation=90)
        ax[z].set_title(k)
        ax[z].set_xticks(alphas)
        ax[z].set_xticklabels(['{:.2f}'.format(alpha) for alpha in alphas])
        ax[z].set_xlabel(r'$\alpha$ [-]')
        if z > 0:
            ax[z].set_yticklabels([])
        z += 1

    uniform_lims(ax, 'y')
    plt.sca(ax[0])
    ax[0].set_ylabel(r'$r_{\tau_i}$ [-]')
    add_colorbar(cm, 1, 24, 'step ahead [h]')

    plt.savefig('figs/{}.pdf'.format(name))


def get_dists_from_xy(x,y):
    p = (x+y)/2
    d = ((p-x)**2 + (p-y)**2)**0.5
    return d


def plot_reliability_tilted(results,size,font_scale):
    name = 'reliability_tilted'
    n_res = len(results)
    f,ax = set_figure(size=size,subplots=(1,n_res),w=0.05, h=0,b=0.2,font_scale=font_scale)
    z = 0
    n_q = results[list(results.keys())[0]]['reliability'].shape[0]
    alphas = np.linspace(1 / n_q, 1 - 1 / n_q, n_q)
    for k,v in results.items():
        plt.sca(ax[z])
        n = v['reliability'].shape[1]
        cm = plt.get_cmap('viridis',n)
        for i in range(n):
            d = alphas - v['reliability'][:, i]
            plt.plot(alphas, d, label=k, c = cm(i), alpha=0.8)
        plt.xticks(rotation=90)
        ax[z].set_title(k)
        ax[z].set_xticks(alphas)
        ax[z].set_xticklabels(['{:.2f}'.format(alpha) for alpha in alphas])
        ax[z].set_xlabel(r'$\alpha$ [-]')
        if z > 0:
            ax[z].set_yticklabels([])
        z += 1

    uniform_lims(ax, 'y')
    plt.sca(ax[0])
    ax[0].set_ylabel(r'$\hat{\alpha}$ [-]')
    add_colorbar(cm, 1, 24, 'step ahead [h]')
    plt.savefig('figs/{}.pdf'.format(name))


def plot_reliability_diff(results,size,font_scale):
    name = 'reliability_diff_tilted'
    n_res = len(results)
    f,ax = set_figure(size=size,subplots=(1,n_res-1),w=0.05, h=0,b=0.2,font_scale=font_scale)
    z = 0
    n_q = results[list(results.keys())[0]]['reliability'].shape[0]
    alphas = np.linspace(1 / n_q, 1 - 1 / n_q, n_q)
    for k,v in results.items():
        if k=='mimo':
            continue
        plt.sca(ax[z])
        n = v['reliability'].shape[1]
        cm = plt.get_cmap('viridis',n)
        for i in range(n):
            d = np.abs(alphas - v['reliability'][:, i])
            d_mimo = np.abs(alphas - results['mimo']['reliability'][:, i])
            plt.plot(alphas, d_mimo-d, label=k, c = cm(i), alpha=0.8)
        plt.xticks(rotation=90)
        ax[z].set_title(k)
        ax[z].set_xticks(alphas)
        ax[z].set_xticklabels(['{:.2f}'.format(alpha) for alpha in alphas])
        ax[z].set_xlabel(r'$\alpha$ [-]')
        ax[z].hlines(0,0,1,linestyle='--',color='grey')
        if z > 0:
            ax[z].set_yticklabels([])
        z += 1

    uniform_lims(ax, 'y')
    plt.sca(ax[0])
    ax[0].set_ylabel(r'$Rs$ [-]')
    add_colorbar(cm, 1, 24, 'step ahead [h]')
    plt.savefig('figs/{}.pdf'.format(name))


def plot_QS(results,size,font_scale):
    name = 'QS'
    n_res = len(results)
    f,ax = set_figure(size=size,subplots=(1,n_res),w=0.05, h=0,b=0.2,font_scale=font_scale)
    z = 0
    n_q = results[list(results.keys())[0]]['reliability'].shape[0]
    alphas = np.linspace(1 / n_q, 1 - 1 / n_q, n_q)
    for k,v in results.items():
        plt.sca(ax[z])
        n = v['skill'].shape[1]
        cm = plt.get_cmap('viridis',n)
        for i in range(n):
            d = v['skill'][:, i]
            plt.plot(alphas, d, label=k, c = cm(i), alpha=0.8)
        plt.xticks(rotation=90)
        ax[z].set_title(k)
        ax[z].set_xticks(alphas)
        ax[z].set_xticklabels(['{:.2f}'.format(alpha) for alpha in alphas])
        ax[z].set_xlabel(r'$\alpha$ [-]')
        if z>0:
            ax[z].set_yticklabels([])
        z += 1

    uniform_lims(ax,'y')


    plt.sca(ax[0])
    ax[0].set_ylabel(r'$\bar{l}_q$ [kW]')
    add_colorbar(cm, 1, 24, 'step ahead [h]')
    plt.savefig('figs/{}.pdf'.format(name))


def plot_QS_diff(results,size,font_scale):
    name = 'QS_diff'
    n_res = len(results)
    f,ax = set_figure(size=size,subplots=(1,n_res-1),w=0.05, h=0,b=0.2,font_scale=font_scale)
    z = 0
    n_q = results[list(results.keys())[0]]['reliability'].shape[0]
    alphas = np.linspace(1 / n_q, 1 - 1 / n_q, n_q)
    for k,v in results.items():
        if k=='mimo':
            continue
        plt.sca(ax[z])
        n = v['skill'].shape[1]
        cm = plt.get_cmap('viridis',n)
        for i in range(n):
            d_mimo = results['mimo']['skill'][:, i]
            d = v['skill'][:, i]
            plt.plot(alphas, d_mimo-d, label=k, c = cm(i), alpha=0.8)
        plt.xticks(rotation=90)
        ax[z].set_title(k)
        ax[z].set_xticks(alphas)
        ax[z].set_xticklabels(['{:.2f}'.format(alpha) for alpha in alphas])
        ax[z].set_xlabel(r'$\alpha$ [-]')
        ax[z].hlines(0, 0, 1, linestyle='--', color='grey')
        if z>0:
            ax[z].set_yticklabels([])
        z += 1

    uniform_lims(ax,'y')


    plt.sca(ax[0])
    ax[0].set_ylabel(r'$\Delta \bar{l}_q$ [kW]')
    add_colorbar(cm, 1, 24, 'step ahead [h]')
    plt.savefig('figs/{}.pdf'.format(name))


def uniform_lims(ax,coord):
    if coord == 'x':
        x_lims = []
        for a in ax:
            x_lims.append(np.array(a.get_xlim()))
        x_lims = (np.min(np.vstack(x_lims)[:,0]),np.max(np.vstack(x_lims)[:,1]))
        for a in ax:
            a.set_xlim(x_lims)
    elif coord == 'y':
        y_lims = []
        for a in ax:
            y_lims.append(np.array(a.get_ylim()))
        y_lims = (np.min(np.vstack(y_lims)[:,0]),np.max(np.vstack(y_lims)[:,1]))
        for a in ax:
            a.set_ylim(y_lims)


def add_colorbar(cm,scale_min, scale_max, label):
    plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)
    cax = plt.axes([0.88, 0.1, 0.025, 0.8])
    sm = plt.cm.ScalarMappable(cmap=cm)
    cb = plt.colorbar(mappable=sm, cax=cax)
    cb.set_label(label)
    cb.ax.set_yticklabels(['{}'.format(i) for i in
                           np.linspace(scale_min, scale_max,
                                       6,dtype=int)], rotation=90)
