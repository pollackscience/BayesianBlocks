#! /usr/bin/env python3

import numpy as np


def rough(hist):
    '''Determine the roughness of a histogram by calculating the 2nd order forward finite diff'''

    total = np.sum(hist[0])
    widths = np.diff(hist[1])
    cd = (widths[:-1]+widths[1:])/2
    cd2 = cd[:-1]+cd[1:]
    f = hist[0]/(total*widths)
    fdp = (f[2:]-2*f[1:-1]+f[:-2])/(cd[:-1]*cd[1:])
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
