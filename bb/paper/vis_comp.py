#! /usr/bin/env python3

import os
import six.moves.cPickle as pkl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from skhep.visual import MplPlotter as skh_plt
from skhep.modeling import bayesian_blocks
from astropy.stats import knuth_bin_width
from utilities import rough, err_nn, err_li, bep_optimizer, normalized, bb_optimizer
from adaptivekde.sshist import sshist


def avg_eli():
    bb_dir = os.path.join('/Users/brianpollack/Coding/BayesianBlocks')
    # xlims = (50, 140)
    # ratlims = (0, 2.5)
    n_events = 10000

    z_data = pd.read_pickle(bb_dir+'/files/DY/ZLL_v2.p')
    z_data = z_data.query('50<Mll<140')
    init_data = z_data[0:n_events].Mll.values

    plt.close()
    np.random.seed(10101)
    # init_data = np.random.uniform(size=10000)
    hist = skh_plt.hist(init_data, 'blocks', p0=0.5)
    # hist = skh_plt.hist(init_data, 'fd')
    plt.show()
    # eli = err_li(init_data, hist)
    elis = []
    for i in range(1, int(len(z_data)/n_events)):
        data = z_data[n_events*i:n_events*(i+1)].Mll.values
        elis.append(err_li(data, hist))
    # np.mean(elis)
    print(np.mean(elis))


def comp_study(input_data, n_events, p0, xlims=None):

    data_nom = input_data[:n_events]
    fig, axes = plt.subplots(3, 3, sharex=True, sharey=False)

    axes[0][0].set_title('Sturges')
    hist_sturges_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='sturges',
                                   errorbars=False, alpha=1, log=True,
                                   scale='binwidth', err_type='gaussian', ax=axes[0][0])

    axes[0][1].set_title('Doane')
    hist_sturges_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='doane',
                                   errorbars=False, alpha=1, log=True,
                                   scale='binwidth', err_type='gaussian', ax=axes[0][0])

    axes[0][2].set_title('Scott')
    hist_scott_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='scott', errorbars=False,
                                 alpha=1, log=True, scale='binwidth', err_type='gaussian',
                                 ax=axes[0][1])

    axes[1][0].set_title('Freedman Diaconis')
    hist_fd_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='fd', errorbars=False,
                              alpha=1, log=True, scale='binwidth',
                              err_type='gaussian', ax=axes[1][0])

    axes[1][1].set_title('Knuth')
    _, bk = knuth_bin_width(data_nom, return_bins=True)
    hist_knuth_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins=bk, errorbars=False,
                                 alpha=1, log=True, scale='binwidth', err_type='gaussian',
                                 ax=axes[0][2])

    axes[1][2].set_title('Shimazaki')
    bshim = sshist(data_nom)
    hist_sturges_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins=bshim[0],
                                   errorbars=False, alpha=1, log=True,
                                   scale='binwidth', err_type='gaussian', ax=axes[0][0])

    axes[2][0].set_title(r'$\sqrt(n)$')
    hist_sqrt_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='sqrt', errorbars=False,
                                alpha=1, log=True, scale='binwidth',
                                err_type='gaussian', ax=axes[1][1])


    # bep = bep_optimizer(data_nom)
    # _, bep = pd.qcut(data_nom, nep, retbins=True)


    hist_sturges = np.histogram(data_nom, bins='sturges')
    hist_doane = np.histogram(data_nom, bins='doane')
    hist_scott = np.histogram(data_nom, bins='scott')
    hist_fd = np.histogram(data_nom, bins='fd')
    hist_knuth = np.histogram(data_nom, bins=bk)
    hist_shim = np.histogram(data_nom, bins=bshim[0])
    hist_sqrt = np.histogram(data_nom, bins='sqrt')

    r_sturges = rough(hist_sturges_bw)
    r_doane = rough(hist_doane_bw)
    r_scott = rough(hist_scott_bw)
    r_fd = rough(hist_fd_bw)
    r_knuth = rough(hist_knuth_bw)
    r_shim = rough(hist_shim_bw)
    r_sqrt = rough(hist_sqrt_bw)

    eli_sturges = err_li(data_nom, hist_sturges)
    eli_doane = err_li(data_nom, hist_doane)
    eli_scott = err_li(data_nom, hist_scott)
    eli_fd = err_li(data_nom, hist_fd)
    eli_knuth = err_li(data_nom, hist_knuth)
    eli_shim = err_li(data_nom, hist_shim)
    eli_sqrt = err_li(data_nom, hist_sqrt)

    avg_eli_sturges = []
    avg_eli_doane = []
    avg_eli_scott = []
    avg_eli_knuth = []
    avg_eli_fd = []
    avg_eli_sqrt = []
    for i in range(1, min(int(len(input_data)/n_events), 100)):
        data = input_data[n_events*i:n_events*(i+1)]
        avg_eli_sturges.append(err_li(data, hist_sturges))
        avg_eli_scott.append(err_li(data, hist_scott))
        avg_eli_knuth.append(err_li(data, hist_knuth))
        avg_eli_fd.append(err_li(data, hist_fd))
        avg_eli_sqrt.append(err_li(data, hist_sqrt))

    avg_eli_sturges = np.mean(avg_eli_sturges)
    avg_eli_scott = np.mean(avg_eli_scott)
    avg_eli_knuth = np.mean(avg_eli_knuth)
    avg_eli_fd = np.mean(avg_eli_fd)
    avg_eli_sqrt = np.mean(avg_eli_sqrt)

    eli_lims = [avg_eli_sturges, avg_eli_scott, avg_eli_knuth, avg_eli_fd, avg_eli_sqrt]
    r_lims = [r_sturges, r_scott, r_knuth, r_fd, r_sqrt]

    axes[1][2].set_title('Bayesian Blocks')
    p0 = bb_optimizer(data_nom, min(int(len(input_data)/n_events), 100), input_data[n_events:],
                      r_lims, eli_lims)
    bb = bayesian_blocks(data_nom, p0=p0)
    if xlims:
        bb[0] = xlims[0]
        bb[-1] = xlims[-1]
    hist_bb_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins=bb, errorbars=False,
                              alpha=1, log=True, scale='binwidth',
                              err_type='gaussian', ax=axes[1][2])

    hist_bb = np.histogram(data_nom, bins=bb)
    r_bb = rough(hist_bb_bw)
    eli_bb = err_li(data_nom, hist_bb)
    avg_eli_bb = []
    for i in range(1, min(int(len(input_data)/n_events), 100)):
        data = input_data[n_events*i:n_events*(i+1)]
        avg_eli_bb.append(err_li(data, hist_bb))
    avg_eli_bb = np.mean(avg_eli_bb)

    rs = [r_sturges, r_scott, r_knuth, r_fd, r_sqrt, r_bb]
    elis = [eli_sturges, eli_scott, eli_knuth, eli_fd, eli_sqrt, eli_bb]
    avg_elis = [avg_eli_sturges, avg_eli_scott, avg_eli_knuth, avg_eli_fd, avg_eli_sqrt, avg_eli_bb]
    print('eli:', avg_eli_bb, 'rough:', r_bb)

    rs_mod = normalized(np.asarray(rs)**(1/3))
    avg_elis_mod = normalized(-np.asarray(avg_elis)**(-2))
    xs = ['doane', 'scott', 'knuth', 'fd', 'sqrt', 'bb']

    plt.figure()
    plt.plot(xs, rs, 'o-', label='unmodified')
    ax2 = plt.gca().twinx()
    ax2.plot(xs, rs_mod, 'o-', color='C2', label='linearized')
    plt.legend()
    plt.grid()
    plt.title('Roughness')

    plt.figure()
    plt.plot(xs, elis, 'o-', label='same set')
    plt.plot(xs, avg_elis, 's-', label='average sets')
    ax2 = plt.gca().twinx()
    ax2.plot(xs, avg_elis_mod, 's-', color='C2', label='linearized')
    plt.legend()
    plt.grid()
    plt.title('L1 Norm (lin interp)')

    plt.figure()
    for i in range(len(elis)):
        plt.scatter(avg_elis[i], rs[i], label=xs[i], s=200)
    plt.xlabel('Average err_LI')
    plt.ylabel('Roughness')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.relim()
    # ax.autoscale_view()
    plt.grid()
    plt.legend()
    plt.title('2D Score (unmodified)')

    plt.figure()
    for i in range(len(elis)):
        plt.scatter(avg_elis_mod[i], rs_mod[i], label=xs[i], s=200)
    plt.xlabel('Average err_LI')
    plt.ylabel('Roughness')
    plt.grid()
    plt.legend()
    plt.title('2D Score (linearized)')

    plt.figure()
    plt.plot(xs, normalized(avg_elis)+normalized(rs), 'o-', label='unmodified')
    print((normalized(avg_elis)+normalized(rs)))
    plt.plot(xs, normalized(-np.asarray(avg_elis)**(-2))+normalized(np.asarray(rs)**(1/3)), 'o-',
             label='linearized')
    plt.grid()
    plt.legend()
    plt.title('Combined Score')


if __name__ == "__main__":
    avg_eli()
