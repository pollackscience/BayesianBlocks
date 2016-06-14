#! /usr/bin/env python

from __future__ import division
import os
import functools
import numpy as np
import cPickle as pkl
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.stats import poisson
from lmfit import Model
from hgg_comp import generate_initial_params, generate_toy_data, bg_pdf, sig_pdf

def lm_binned_wrapper(mu_bg, mu_sig):
    def lm_binned(ix, A, ntot):
        proper_means = ((mu_bg+A*mu_sig)/np.sum(mu_bg+A*mu_sig))*ntot
        return proper_means[ix]
    return lm_binned

if __name__ == "__main__":

    plt.close('all')
    current_dir = os.path.dirname(__file__)
    bb_dir      = os.path.join(current_dir, '../..')
    hgg_bg      = pkl.load(open(bb_dir+'/files/hgg_bg.p', "rb"))
    hgg_signal  = pkl.load(open(bb_dir+'/files/hgg_signal.p', "rb"))

    bg_result, sig_result, n_bg, n_sig, be_bg, be_sig = generate_initial_params(hgg_bg, hgg_signal, 5)
    be_hybrid = np.concatenate([be_bg[be_bg<be_sig[0]-1.5], be_sig, be_bg[be_bg>be_sig[-1]+1.5]])
    be_1GeV = np.linspace(100,180,81)
    be_2GeV = np.linspace(100,180,41)
    be_5GeV = np.linspace(100,180,17)
    be_10GeV = np.linspace(100,180,9)

    true_bg_bc  = []
    true_sig_bc = []
    for i in range(len(be_hybrid)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),be_hybrid[i],be_hybrid[i+1])
        true_bg_bc.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),be_hybrid[i],be_hybrid[i+1])
        true_sig_bc.append(true_sig*n_sig)


    lm_binned = lm_binned_wrapper(np.asarray(true_bg_bc), np.asarray(true_sig_bc))
    binned_model = Model(lm_binned)
    binned_params = binned_model.make_params()
    binned_params['ntot'].value   = n_bg+n_sig
    binned_params['ntot'].vary    = False
    binned_params['A'].value      = 0.1
    binned_params['A'].min        = 0
    binned_params['A'].max        = 1000

    mc_bg, mc_sig = generate_toy_data(bg_result.x, sig_result.x, n_bg, n_sig)
    mc_bg_sig = mc_bg+mc_sig
    bc, _ = np.histogram(mc_bg_sig, be_hybrid, range=(100,180))
    ibc = np.asarray(range(len(bc)))
    result = binned_model.fit(bc, ix = ibc, params = binned_params)
    nll_bg = -np.sum(np.log(poisson.pmf(bc,result.eval(A=0))))
    nll_sig = -np.sum(np.log(poisson.pmf(bc,result.best_fit)))
    q0=2*(nll_bg-nll_sig)



