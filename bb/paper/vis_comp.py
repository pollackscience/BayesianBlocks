#! /usr/bin/env python3

import os
import six.moves.cPickle as pkl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from skhep.visual import MplPlotter as skh_plt
from astropy.stats import knuth_bin_width
from utilities import rough, err_nn, err_li


def mini_study():
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


if __name__ == "__main__":
    mini_study()
