#! /usr/bin/env python3

import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks
from scipy import interpolate
from matplotlib import pyplot as plt
from scipy.stats import rankdata


def rough(hist, plot=False):
    '''Determine the roughness of a histogram by calculating the 2nd order forward finite diff'''

    heights = hist[0]
    widths = np.diff(hist[1])
    coords = hist[1][:-1]+widths/2
    # f = interpolate.InterpolatedUnivariateSpline(coords, heights, k=1)
    # coords_new = np.linspace(coords[0], coords[-1], 10000)
    # widths_eff = (coords[-1]-coords[0])/10000
    # heights_new = f(coords_new)
    # ff = interpolate.InterpolatedUnivariateSpline(coords_new, heights_new, k=3)
    # fd1 = f.derivative(1)
    # dh = np.diff(coords)
    # cd = np.concatenate([[0], (widths[:-1]+widths[1:])/2])
    # cd2 = cd[:-1]+cd[1]
    # f = np.log(hist[0]+1e-15)
    # fdp = (f[2:]-2*f[:-1]+f[:-2])/(cd[:-1]*cd[1:])
    # fdp = cd[:1](f[2:]-2*f[1:-1]+f[:-2])/(cd[:-1]*cd[1:])
    # fdp = (f[2:]-2*f[1:-1]+f[:-2])/(widths[:-2]*widths[2:])
    # print()
    # print(widths)
    fd1 = np.gradient(heights, coords, edge_order=1)
    sfd1 = np.sign(fd1)
    # print(fd1)
    # fd2 = np.gradient(fd1, coords, edge_order=1)
    # priHnt(fd2)

    # rough = np.trapz(fd2**2, coords)
    rough = np.unique(sfd1[:-1]*sfd1[1:], return_counts=True)[1][0]
    # print(rough)
    # if plot:
    #     plt.figure()
    #     print(coords)
    #     print(heights)
    #     plt.plot(coords, heights, 'ro-', ms=5)
    #     # plt.plot(coords_new, f(coords_new), 'g', lw=3, alpha=0.7)
    #     plt.figure()
    #     plt.plot(coords, fd2, 'g', lw=3, alpha=0.7)

    # print(rough)
    return rough


def err_nn(input_data, hist):
    cd = (hist[1][:-1]+hist[1][1:])/2
    counts = hist[0]
    nn_data = [[c]*int(n) for c, n in zip(cd, counts)]
    nn_data = [item for sublist in nn_data for item in sublist]
    # print(list(zip(np.asarray(nn_data), np.sort(input_data))))
    return np.sum(np.abs(np.asarray(nn_data)-np.sort(input_data)))


def err_li(input_data, hist):
    counts = hist[0]
    nn_data = []
    for i in range(len(hist[1])-1):
        if counts[i] == 1:
            nn_data.append(np.asarray([(hist[1][i]+hist[1][i+1])/2]))
        else:
            nn_data.append(np.linspace(hist[1][i], hist[1][i+1], counts[i]))
    nn_data = np.concatenate(nn_data)
    # print(list(zip(np.asarray(nn_data), np.sort(input_data))))
    return np.sum(np.abs(np.asarray(nn_data)-np.sort(input_data)))


def avg_err_li(hist, n_events, n_resamples, input_data=None, input_dist=None,
               sample_dict=None):
    np.random.seed(100)
    if not (np.all(input_data) or input_dist):
        raise ValueError('input_data or input_dist must be defined')
    if (np.all(input_data) and input_dist):
        raise ValueError('only input_data or input_dist can be defined')

    err_li_list = []
    if np.all(input_data):
        for i in range(n_resamples-1):
            err_li_list.append(err_li(np.random.choice(input_data, size=n_events), hist))

    elif input_dist:
        for i in range(n_resamples-1):
            err_li_list.append(err_li(input_dist(size=n_events), hist))

    return np.mean(err_li_list)


def bep_optimizer(data, resample_list, roughs, elis):
    best_rank = np.inf
    best_bep = 0
    for nb in range(10, int(np.sqrt(len(data)))):
        _, bep = pd.qcut(data, nb, retbins=True)
        tmp_hist = np.histogram(data, bins=bep)
        tmp_hist_bw = tmp_hist[0]/np.diff(tmp_hist[1])
        tmp_rough = rough((tmp_hist_bw, tmp_hist[1]))
        tmp_eli = []
        for i in resample_list:
            tmp_eli.append(err_li(i, tmp_hist))
        tmp_eli = np.mean(tmp_eli)

        rank_rough = rankdata(roughs + [tmp_rough])
        rank_eli = rankdata(elis + [tmp_eli])
        tmp_rank = rank_eli[-1] + rank_rough[-1]
        # print(p0, tmp_metric)

        if tmp_rank < best_rank:
            best_bep = bep
            best_rank = tmp_rank
    return best_bep


def bb_optimizer(data, resample_list, roughs, elis):
    best_rank = np.inf
    best_p0 = 0
    p0s = np.logspace(-4.5, 0, 50)
    for p0 in p0s:
        bb = bayesian_blocks(data, p0=p0)
        tmp_hist = np.histogram(data, bins=bb)
        tmp_hist_bw = tmp_hist[0]/np.diff(tmp_hist[1])
        tmp_rough = rough((tmp_hist_bw, tmp_hist[1]))
        tmp_eli = []
        for i in resample_list:
            tmp_eli.append(err_li(i, tmp_hist))
        tmp_eli = np.mean(tmp_eli)

        rank_rough = rankdata(roughs + [tmp_rough])
        rank_eli = rankdata(elis + [tmp_eli])
        tmp_rank = rank_eli[-1] + rank_rough[-1]
        # print(p0, tmp_metric)

        if tmp_rank <= best_rank:
            best_p0 = p0
            best_rank = tmp_rank
    return best_p0


def normalized(a):
    a = np.asarray(a)
    norm1 = (a - min(a))/(max(a)-min(a))
    return norm1
