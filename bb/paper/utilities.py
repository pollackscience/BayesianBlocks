#! /usr/bin/env python3

import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks


def rough(hist):
    '''Determine the roughness of a histogram by calculating the 2nd order forward finite diff'''

    widths = np.diff(hist[1])
    cd = (widths[:-1]+widths[1:])/2
    cd2 = cd[:-1]+cd[1:]
    # f = np.log(hist[0]+1e-15)
    f = hist[0]
    fdp = (f[2:]-2*f[1:-1]+f[:-2])/(cd[:-1]*cd[1:])
    # fdp = (f[2:]-2*f[1:-1]+f[:-2])/(widths[:-2]*widths[2:])
    rough = np.sum(fdp**2*cd2)
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


def avg_err_li(hist, n_events, n_resamples, input_data=None, input_dist=None):
    if not (np.all(input_data) or input_dist):
        raise ValueError('input_data or input_dist must be defined')
    if (np.all(input_data) and input_dist):
        raise ValueError('only input_data or input_dist can be defined')

    err_li_list = []
    if np.all(input_data):
        for i in range(n_resamples-1):
            err_li_list.append(err_li(input_data[i*n_events:n_events*(i+1)], hist))

    elif input_dist:
        for i in range(n_resamples-1):
            err_li_list.append(err_li(input_dist(size=n_events), hist))

    return np.mean(err_li_list)


def bep_optimizer(data, n_resamples, resample_data, roughs, elis):
    best_metric = np.inf
    best_bep = 0
    for nb in range(3, int(np.sqrt(len(data)))):
        _, bep = pd.qcut(data, nb, retbins=True)
        tmp_hist = np.histogram(data, bins=bep)
        tmp_hist_bw = tmp_hist[0]/np.diff(tmp_hist[1])
        tmp_rough = rough((tmp_hist_bw, tmp_hist[1]))
        tmp_eli = avg_err_li(tmp_hist, len(data), n_resamples, input_data=resample_data)

        norm_rough = normalized(roughs + [tmp_rough])
        norm_eli = normalized(elis + [tmp_eli])
        tmp_metric = norm_eli[-1] + norm_rough[-1]
        # print(p0, tmp_metric)

        if tmp_metric < best_metric:
            best_bep = bep
            best_metric = tmp_metric
    return best_bep


def bb_optimizer(data, n_resamples, resample_data, roughs, elis):
    best_metric = np.inf
    best_p0 = 0
    p0s = np.logspace(-3, 0, 25)
    for p0 in p0s:
        bb = bayesian_blocks(data, p0=p0)
        tmp_hist = np.histogram(data, bins=bb)
        tmp_hist_bw = tmp_hist[0]/np.diff(tmp_hist[1])
        tmp_rough = rough((tmp_hist_bw, tmp_hist[1]))
        tmp_eli = avg_err_li(tmp_hist, len(data), n_resamples, input_data=resample_data)

        norm_rough = normalized(roughs + [tmp_rough])
        norm_eli = normalized(elis + [tmp_eli])
        tmp_metric = norm_eli[-1] + norm_rough[-1]
        # print(p0, tmp_metric)

        if tmp_metric < best_metric:
            best_p0 = p0
            best_metric = tmp_metric
    return best_p0


def normalized(a):
    norm1 = (a - min(a))/(max(a)-min(a))
    return norm1
