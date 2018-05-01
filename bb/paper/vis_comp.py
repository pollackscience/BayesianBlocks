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
from utilities import rough, err_li, avg_err_li, bep_optimizer, normalized, bb_optimizer
from collections import OrderedDict
from scipy.stats import rankdata
from itertools import cycle
from scipy import stats


def comp_study(input_data, n_events, xlims=None, resamples=100, dist_name='2Gauss'):
    bb_dir = os.path.join('/Users/brianpollack/Coding/BayesianBlocks')
    do_log = True

    # data_nom = input_data[:n_events]
    if dist_name == 'Gauss':
        np.random.seed(88)
        data_nom = np.random.normal(125, 2, size=n_events)
        resample_list = np.random.normal(125, 2, size=(resamples, n_events))
        do_log = False

    elif dist_name == '2LP':
        np.random.seed(33)
        data_nom = np.concatenate((np.random.laplace(loc=90, scale=5, size=int(n_events*0.65)),
                                  np.random.laplace(loc=110, scale=1.5, size=int(n_events*0.25)),
                                  np.random.uniform(low=80, high=120, size=int(n_events*0.10))))
        resample_list = np.concatenate((
            np.random.laplace(loc=90, scale=5, size=(resamples, int(n_events*0.65))),
            np.random.laplace(loc=110, scale=1.5, size=(resamples, int(n_events*0.25))),
            np.random.uniform(low=80, high=120, size=(resamples, int(n_events*0.10)))), axis=1)
        do_log = False

    elif dist_name == 'jPT':
        np.random.seed(11)
        data_nom = np.random.choice(input_data, size=n_events, replace=False)
        resample_list = np.random.choice(input_data, size=(resamples, n_events), replace=True)

    elif dist_name == 'DY':
        np.random.seed(200)
        data_nom = np.random.choice(input_data, size=n_events, replace=False)
        resample_list = np.random.choice(input_data, size=(resamples, n_events), replace=True)
    else:
        np.random.seed(1)
        data_nom = np.random.choice(input_data, size=n_events, replace=False)
        resample_list = np.random.choice(input_data, size=(resamples, n_events), replace=True)

    fig_hist, axes_hist = plt.subplots(3, 3, sharex=True, sharey=False, constrained_layout=True)
    fig_hist.suptitle(f'{dist_name} Distribution, N={n_events}', fontsize=22)
    # fig_hist.text(-0.03, 0.5, 'Entries/Bin Width', va='center', rotation='vertical', fontsize=20)
    # axes_hist[2][0].get_xaxis().set_ticks([])
    # axes_hist[2][1].get_xaxis().set_ticks([])
    # axes_hist[2][2].get_xaxis().set_ticks([])

    axes_hist[0][0].set_title('Sturges')
    hist_sturges_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='sturges',
                                   errorbars=False, alpha=0.5, log=do_log,
                                   scale='binwidth', err_type='gaussian', ax=axes_hist[0][0])

    axes_hist[0][1].set_title('Doane')
    hist_doane_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='doane',
                                 errorbars=False, alpha=0.5, log=do_log,
                                 scale='binwidth', err_type='gaussian', ax=axes_hist[0][1])

    axes_hist[0][2].set_title('Scott')
    hist_scott_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='scott',
                                 errorbars=False, alpha=0.5, log=do_log, scale='binwidth',
                                 err_type='gaussian', ax=axes_hist[0][2])

    axes_hist[1][0].set_title('Freedman Diaconis')
    axes_hist[1][0].set_ylabel('Entries/Bin Width', fontsize=20)
    hist_fd_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='fd', errorbars=False,
                              alpha=0.5, log=do_log, scale='binwidth',
                              err_type='gaussian', ax=axes_hist[1][0])

    axes_hist[1][1].set_title('Knuth')
    _, bk = knuth_bin_width(data_nom, return_bins=True)
    hist_knuth_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins=bk, errorbars=False,
                                 alpha=0.5, log=do_log, scale='binwidth', err_type='gaussian',
                                 ax=axes_hist[1][1])

    axes_hist[1][2].set_title('Rice')
    hist_rice_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='rice',
                                errorbars=False, alpha=0.5, log=do_log,
                                scale='binwidth', err_type='gaussian', ax=axes_hist[1][2])

    axes_hist[2][0].set_title('Sqrt(N)')
    hist_sqrt_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins='sqrt', errorbars=False,
                                alpha=0.5, log=do_log, scale='binwidth',
                                err_type='gaussian', ax=axes_hist[2][0])

    # bep = bep_optimizer(data_nom)
    # _, bep = pd.qcut(data_nom, nep, retbins=True)

    hist_sturges = np.histogram(data_nom, bins='sturges')
    hist_doane = np.histogram(data_nom, bins='doane')
    hist_scott = np.histogram(data_nom, bins='scott')
    hist_fd = np.histogram(data_nom, bins='fd')
    hist_knuth = np.histogram(data_nom, bins=bk)
    hist_rice = np.histogram(data_nom, bins='rice')
    hist_sqrt = np.histogram(data_nom, bins='sqrt')

    r_sturges = rough(hist_sturges_bw, plot=False)
    r_doane = rough(hist_doane_bw)
    r_scott = rough(hist_scott_bw)
    r_fd = rough(hist_fd_bw)
    r_knuth = rough(hist_knuth_bw, plot=False)
    r_rice = rough(hist_rice_bw)
    r_sqrt = rough(hist_sqrt_bw, plot=False)

    eli_sturges = err_li(data_nom, hist_sturges)
    eli_doane = err_li(data_nom, hist_doane)
    eli_scott = err_li(data_nom, hist_scott)
    eli_fd = err_li(data_nom, hist_fd)
    eli_knuth = err_li(data_nom, hist_knuth)
    eli_rice = err_li(data_nom, hist_rice)
    eli_sqrt = err_li(data_nom, hist_sqrt)

    avg_eli_sturges = []
    avg_eli_doane = []
    avg_eli_scott = []
    avg_eli_fd = []
    avg_eli_knuth = []
    avg_eli_rice = []
    avg_eli_sqrt = []
    for i in resample_list:
        avg_eli_sturges.append(err_li(i, hist_sturges))
        avg_eli_doane.append(err_li(i, hist_doane))
        avg_eli_scott.append(err_li(i, hist_scott))
        avg_eli_fd.append(err_li(i, hist_fd))
        avg_eli_knuth.append(err_li(i, hist_knuth))
        avg_eli_rice.append(err_li(i, hist_rice))
        avg_eli_sqrt.append(err_li(i, hist_sqrt))

    avg_eli_sturges = np.mean(avg_eli_sturges)
    avg_eli_doane = np.mean(avg_eli_doane)
    avg_eli_scott = np.mean(avg_eli_scott)
    avg_eli_fd = np.mean(avg_eli_fd)
    avg_eli_knuth = np.mean(avg_eli_knuth)
    avg_eli_rice = np.mean(avg_eli_rice)
    avg_eli_sqrt = np.mean(avg_eli_sqrt)

    avg_eli_list = [avg_eli_sturges, avg_eli_doane, avg_eli_scott,
                    avg_eli_fd, avg_eli_knuth, avg_eli_rice, avg_eli_sqrt]
    r_list = [r_sturges, r_doane, r_scott, r_fd, r_knuth, r_rice, r_sqrt]

    elis_list = [eli_sturges, eli_doane, eli_scott, eli_fd, eli_knuth, eli_rice, eli_sqrt]

    axes_hist[2][1].set_title('Equal Population')
    bep = bep_optimizer(data_nom, resample_list, r_list, avg_eli_list)
    hist_ep_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins=bep, errorbars=False,
                              alpha=0.5, log=do_log, scale='binwidth',
                              err_type='gaussian', ax=axes_hist[2][1])
    hist_ep = np.histogram(data_nom, bins=bep)
    r_ep = rough(hist_ep_bw)
    eli_ep = err_li(data_nom, hist_ep)
    avg_eli_ep = []
    for i in resample_list:
        avg_eli_ep.append(err_li(i, hist_ep))
    avg_eli_ep = np.mean(avg_eli_ep)

    axes_hist[2][2].set_title('Bayesian Blocks')
    p0 = bb_optimizer(data_nom, resample_list, r_list, avg_eli_list)
    bb = bayesian_blocks(data_nom, p0=p0)
    if xlims:
        bb[0] = xlims[0]
        bb[-1] = xlims[-1]
    hist_bb_bw = skh_plt.hist(x=data_nom, histtype='stepfilled', bins=bb, errorbars=False,
                              alpha=1, log=do_log, scale='binwidth',
                              err_type='gaussian', ax=axes_hist[2][2])
    # if n_events == 1000 and dist_name == '2LP':
    # axes_hist[2][2].set_ylim((0, 100))
    hist_bb = np.histogram(data_nom, bins=bb)
    r_bb = rough(hist_bb_bw, plot=False)
    eli_bb = err_li(data_nom, hist_bb)
    avg_eli_bb = []
    for i in resample_list:
        avg_eli_bb.append(err_li(i, hist_bb))
    avg_eli_bb = np.mean(avg_eli_bb)

    r_list.append(r_ep)
    r_list.append(r_bb)
    avg_eli_list.append(avg_eli_ep)
    avg_eli_list.append(avg_eli_bb)
    elis_list.append(eli_ep)
    elis_list.append(eli_bb)
    plt.savefig(bb_dir+f'/plots/bin_comp/hists_{dist_name}_{n_events}.pdf')

    xs = ['Sturges', 'Doane', 'Scott', 'FD', 'Knuth', 'Rice', 'Sqrt', 'EP', 'BB']

    fig_metric, axes_metric = plt.subplots(2, 1, constrained_layout=True)
    fig_hist.suptitle(f'{dist_name} Distribution, N={n_events}')
    for i in range(len(elis_list)):
        if xs[i] == 'BB':
            axes_metric[0].scatter(avg_eli_list[i], r_list[i], label=xs[i], s=400,
                                   marker='*', c='k')
        else:
            axes_metric[0].scatter(avg_eli_list[i], r_list[i], label=xs[i], s=200)
    axes_metric[0].set_ylabel(r'$W_n$ (Wiggles)')
    axes_metric[0].set_xlabel(r'$\hat{E}$ (Average Error)')
    # ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.relim()
    # ax.autoscale_view()
    axes_metric[0].grid()
    axes_metric[0].legend(ncol=1, bbox_to_anchor=(1.05, 1.15), loc='upper left')
    axes_metric[0].set_title(f'{dist_name} Distribution, N={n_events}', fontsize=22)
    # plt.savefig(bb_dir+f'/plots/bin_comp/scat_{dist_name}_{n_events}.pdf')

    # plt.figure()
    rank_rough = rankdata(r_list, method='min')
    rank_avg_eli = rankdata(avg_eli_list, method='min')

    cont = axes_metric[1].bar(xs, rank_rough, 0.35, label=r'$W_n$ Ranking', alpha=0.5)
    cont[-1].set_alpha(1)
    cont = axes_metric[1].bar(xs, rank_avg_eli, 0.35, bottom=rank_rough, label=r'$\hat{E}$ Ranking',
                              alpha=0.5)
    cont[-1].set_alpha(1)
    axes_metric[1].legend(loc='upper left', bbox_to_anchor=(1.0, 0.8))
    # axes_metric[1].set_title(f'Combined Ranking, {dist_name} Distribution, N={n_events}')
    axes_metric[1].set_xlabel('Binning Method')
    axes_metric[1].set_ylabel('Rank')
    plt.savefig(bb_dir+f'/plots/bin_comp/metric_{dist_name}_{n_events}.pdf')

    # rs_mod = normalized(np.asarray(r_list)**(1/3))
    # avg_elis_mod = normalized(-np.asarray(avg_eli_list)**(-2))
    # plt.figure()
    # plt.plot(xs, r_list, 'o-', label='unmodified')
    # ax2 = plt.gca().twinx()
    # ax2.plot(xs, rs_mod, 'o-', color='C2', label='linearized')
    # plt.legend()
    # plt.grid()
    # plt.title('Roughness')

    # plt.figure()
    # plt.plot(xs, elis_list, 'o-', label='same set')
    # plt.plot(xs, avg_eli_list, 's-', label='average sets')
    # ax2 = plt.gca().twinx()
    # ax2.plot(xs, avg_elis_mod, 's-', color='C2', label='linearized')
    # plt.legend()
    # plt.grid()
    # plt.title('L1 Norm (lin interp)')

    # plt.figure()
    # for i in range(len(elis_list)):
    #     plt.scatter(avg_elis_mod[i], rs_mod[i], label=xs[i], s=200)
    # plt.xlabel('Average err_LI')
    # plt.ylabel('Roughness')
    # plt.grid()
    # plt.legend()
    # plt.title('2D Score (linearized)')

    # plt.figure()
    # plt.plot(xs, normalized(avg_eli_list)+normalized(r_list), 'o-', label='unmodified')
    # print((normalized(avg_eli_list)+normalized(r_list)))
    # plt.plot(xs, normalized(-np.asarray(avg_eli_list)**(-2)) +
    #          normalized(np.asarray(r_list)**(1/3)), 'o-',
    #          label='linearized')
    # plt.grid()
    # plt.legend()
    # plt.title('Combined Score')


def metric_examples(input_data):
    bb_dir = os.path.join('/Users/brianpollack/Coding/BayesianBlocks')
    np.random.seed(11)

    fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 6))
    for n_size in [500, 10000]:
        data_nom = np.random.choice(input_data, size=n_size, replace=False)
        resample_list = np.random.choice(input_data, size=(100, n_size), replace=True)
        # data_nom = np.random.normal(0, 1, size=n_size)
        # resample_list = np.random.normal(0, 1, size=(100, n_size))
        rough_list = []
        err_list = []
        bins = list(range(4, 200, 2))
        for n in bins:
            hist = np.histogram(data_nom, bins=n)
            rough_list.append(rough(hist))
            avg_eli = []
            for i in resample_list:
                avg_eli.append(err_li(i, hist))
            err_list.append(np.mean(avg_eli))

        axes[0].plot(bins, rough_list, label=f'N={n_size}')
        axes[1].plot(bins, err_list, label=f'N={n_size}')

    axes[0].set_xlabel('Number of Bins')
    axes[0].set_ylabel(r'$W_n$', fontsize=20)
    axes[0].set_title(r'$W_n$ (Wiggle) Metric')
    # axes[0].legend()

    axes[1].set_xlabel('Number of Bins')
    axes[1].set_ylabel(r'$\hat{E}$', fontsize=20)
    axes[1].set_title(r'$\hat{E}$ (Average Error) Metric')
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.savefig(bb_dir+f'/plots/bin_comp/metric_example.pdf')
