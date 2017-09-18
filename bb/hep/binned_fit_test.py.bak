#! /usr/bin/env python

from __future__ import division
import os
import functools
import numpy as np
import cPickle as pkl
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import hgg_comp
from ROOT import TF1
import nllfitter.future_fitter as ff



if __name__ == "__main__":

    plt.close('all')
    current_dir = os.path.dirname(__file__)
    bb_dir      = os.path.join(current_dir, '../..')
    hgg_bg      = pkl.load(open(bb_dir+'/files/hgg_bg.p', "rb"))
    hgg_signal  = pkl.load(open(bb_dir+'/files/hgg_signal.p', "rb"))

    bg_result, sig_result, n_bg, n_sig, be_bg, be_sig = hgg_comp.generate_initial_params(hgg_bg, hgg_signal, 5)
    n_tot = n_bg + n_sig

    # bg dist ROOT
    bg_pdf_ROOT = functools.partial(hgg_comp.bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 100, 180, 3)
    tf1_bg_pdf.SetParameters(*bg_result.x)

    # signal dist
    sig_pdf_ROOT = functools.partial(hgg_comp.sig_pdf, doROOT=True)
    tf1_sig_pdf = TF1("tf1_sig_pdf", sig_pdf_ROOT, 100, 180, 2)
    tf1_sig_pdf.SetParameters(*sig_result.x)
    be_hybrid = np.concatenate([be_bg[be_bg<be_sig[0]-1.5], be_sig, be_bg[be_bg>be_sig[-1]+1.5]])
    be_1GeV = np.linspace(100,180,81)
    be_2GeV = np.linspace(100,180,41)
    be_5GeV = np.linspace(100,180,17)
    be_10GeV = np.linspace(100,180,9)

    true_bg_bc  = []
    true_sig_bc = []
    for i in range(len(be_hybrid)-1):
        true_bg, _   = integrate.quad(functools.partial(hgg_comp.bg_pdf, a=bg_result.x),be_hybrid[i],be_hybrid[i+1])
        true_bg_bc.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(hgg_comp.sig_pdf, a=sig_result.x),be_hybrid[i],be_hybrid[i+1])
        true_sig_bc.append(true_sig*n_sig)


    template_pdf = hgg_comp.template_pdf_wrapper(true_bg_bc, true_sig_bc)
    template_model = ff.Model(template_pdf, ['A', 'ntot'])
    template_model.set_bounds([(0, 1), (n_tot, n_tot)])

    mc_bg, mc_sig = hgg_comp.generate_toy_data(tf1_bg_pdf, tf1_sig_pdf, n_bg, n_sig)
    mc_bg_sig = mc_bg+mc_sig
    bc, _ = np.histogram(mc_bg_sig, be_hybrid, range=(100,180))
    print bc
    template_fitter = ff.NLLFitter(template_model, bc, verbose=True)
    mle_res = template_fitter.fit([0.1, n_tot], calculate_corr=False)
    #nll_bg = -np.sum(np.log(poisson.pmf(bc,result.eval(A=0))))
    #nll_sig = -np.sum(np.log(poisson.pmf(bc,result.best_fit)))
    #q0=2*(nll_bg-nll_sig)



