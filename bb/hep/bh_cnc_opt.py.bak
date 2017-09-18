#! /usr/bin/env python

from __future__ import division
import os
import functools
# from scipy.stats import norm
import cPickle as pkl
from matplotlib import pyplot as plt
import nllfitter.future_fitter as ff
from ROOT import gRandom
from ROOT import TF1
from tqdm import tqdm_notebook
import bh_comp as bh_tools


if __name__ == "__main__":
    plt.close('all')
    bb_dir  = os.path.join(os.path.dirname(__file__), '../..')
    df_data_mul2 = pkl.load(open(bb_dir+'/files/BH/BH_test_data.p', 'rb'))
    data_bg_mul2 = df_data_mul2[df_data_mul2.ST_mul2_BB >= 2800].ST_mul2_BB.values

    df_data_mul8 = pkl.load(open(bb_dir+'/files/BH/BH_paper_data.p', 'rb'))
    data_bg_mul8 = df_data_mul8[df_data_mul8.ST_mul8_BB >= 2800].ST_mul8_BB.values

    bg_result, n_bg, be_bg = bh_tools.generate_initial_params(data_bg_mul2, data_bg_mul8)
    gen_toy = bh_tools.generate_toy_data_wrapper(bg_result.x)

    # Do a bunch of toys
    gRandom.SetSeed(2)

    bg_sig_model = ff.Model(bh_tools.bg_sig_pdf, ['C', 'mu', 'sigma', 'alpha', 'beta', 'gamma'])
    sig_params = [(5350, 1070), (6000, 1200), (6600, 1300),
                  (7150, 1440), (7800, 1500), (8380, 1660)]
    # sig_params = [(4750, 970)]

    sig_pdf_ROOT = functools.partial(bh_tools.sig_pdf, doROOT=True)
    tf1_sig_pdf = TF1("tf1_sig_pdf", sig_pdf_ROOT, 2800, 13000, 2)
    n_toys = 10000
    xlows = range(3200, 7000, 100)

    for i, sig_p in enumerate(tqdm_notebook(sig_params, desc='Signal Model')):

        cache_true = {}
        a95s = []
        for xlow in tqdm_notebook(xlows, desc='X-cut', leave=False):

            cache_fit = {}
            cnc_A_mle = []
            for j in tqdm_notebook(xrange(n_toys), desc='Toys', leave=False):

                # mc_bg = gen_toy(n_bg)
                mc_bg = gen_toy(500)
                # while max(mc_bg) < xlow:
                #    mc_bg = gen_toy(n_bg)

                cA_mle, cache_true, cache_fit = bh_tools.calc_A_cnc(mc_bg, bg_result.x, sig_p, xlow,
                                                                    cache_true, cache_fit)
                # cA_mle, _, _= bh_tools.calc_A_cnc(mc_bg, bg_result.x, sig_p, xlow)
                cnc_A_mle.append(cA_mle)

            a_95 = sorted(cnc_A_mle)[int(0.95*n_toys)+1]
            a95s.append(a_95)

        best_xlow = xlows[a95s.index(min(a95s))]
        print 'signal:', i, 'a95:', min(a95s), 'xlow_cut:', best_xlow
