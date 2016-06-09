#! /usr/bin/env python

from __future__ import division
import os
import functools
import bisect
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy.stats import poisson
import cPickle as pkl
from matplotlib import pyplot as plt
from bb.tools.bayesian_blocks_modified import bayesian_blocks
import nllfitter.future_fitter as ff
from ROOT import gRandom
from ROOT import TF1


def bg_pdf(x, a, xlow=100, xhigh=180, doROOT=False):
    '''3rd order legendre poly, mapped from [-1,1] to [xlow,xhigh].
    mapping: f(t) = -1+((1--1)/(xhigh-xlow))*(t-xlow)
    a[0]: O(x)   free parameter
    a[1]: O(x^2) free parameter
    a[2]: O(x^3) free parameter
    Normalization factor calculated based on x-range
    '''
    if doROOT:
        x = x[0]  # This is a ROOT-compatible hack
    t = -1 + ((1 + 1)/(xhigh - xlow))*(x - xlow)
    return 1/(xhigh-xlow)*(1.0 + a[0]*t + 0.5*a[1]*(3*t**2 - 1) + 0.5*a[2]*(5*t**3 - 3*t))


def sig_pdf(x, a, doROOT=False):
    '''simple gaussian pdf.
    a[0]: mean
    a[1]: sigma
    '''
    if doROOT:
        x = x[0]  # This is a ROOT-compatible hack
    return (1.0/(a[1]*np.sqrt(2*np.pi)))*np.exp(-(x-a[0])**2/(2*a[1]**2))


def bg_sig_pdf(x, a, xlow=100, xhigh=180, doROOT=False):
    '''legendre bg pdf and gaussian signal pdf, with a relative normalization factor.
    a[0]: normalization and signal strength parameter
    a[1]: signal mean
    a[2]: signal sigma
    a[3]: bg O(x)   free parameter
    a[4]: bg O(x^2) free parameter
    a[5]: bg O(x^3) free parameter
    '''
    if doROOT:
        x = x[0]            # this is a ROOT-compatible hack
    b = [a[3], a[4], a[5]]  # this is a ROOT-compatible hack
    c = [a[1], a[2]]        # this is a ROOT-compatible hack
    return (1 - a[0])*bg_pdf(x, b, xlow=xlow, xhigh=xhigh) + a[0]*sig_pdf(x, c)

def find_le(a, x):
    '''Find rightmost value less than or equal to x'''
    i = bisect.bisect_right(a, x)
    if i:
        return i-1,a[i-1]
    raise ValueError

def find_lt(a, x):
    '''Find rightmost value less than x'''
    i = bisect.bisect_left(a, x)
    if i:
        return i-1,a[i-1]
    raise ValueError

def find_ge(a, x):
    '''Find leftmost item greater than or equal to x'''
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i,a[i]
    raise ValueError

def find_gt(a, x):
    '''Find leftmost value greater than x'''
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i,a[i]
    raise ValueError

def get_mismatch_metric(bc_nominal, be_nominal, bc_test, be_test):
    '''Calculate the total mismatch between two binning schemes.
    Pass bin content and bin edges of the two histograms for comparison'''
    metric = 0
    for i in range(len(be_test)-1):
        low_edge_test         = be_test[i]
        hi_edge_test          = be_test[i+1]
        ni1, low_nom_low_test = find_le(be_nominal, low_edge_test)
        hi_nom_low_test       = find_gt(be_nominal, low_edge_test)[1]
        ni2, low_nom_hi_test  = find_lt(be_nominal, hi_edge_test)
        hi_nom_hi_test        = find_ge(be_nominal, hi_edge_test)[1]
        bc_nom_1              = bc_nominal[ni1]
        try:
            bc_nom_2 = bc_nominal[ni2]
        except:
            bc_nom_2 = -1

        if low_edge_test == low_nom_low_test and hi_edge_test == hi_nom_hi_test:
            # low and high edges for nominal and test match, ignore them and go to next bin
            continue
        elif low_nom_low_test == low_nom_hi_test and hi_nom_low_test == hi_nom_hi_test:
            # test bin completely contained in nominal bin (easy case)
            metric += abs(bc_nom_1 - bc_test[i])
        else:
            # test bin overlaps two bg bins (harder case)
            width_test = hi_edge_test - low_edge_test
            low_width_test = hi_nom_low_test - low_edge_test
            hi_width_test = hi_edge_test - hi_nom_low_test
            metric += abs(bc_nom_1-bc_test[i])*(low_width_test/width_test)
            metric += abs(bc_nom_2-bc_test[i])*(hi_width_test/width_test)
    return metric

def get_mismatch_metric_v2(bc_nominal, be_nominal, bc_test, be_test, main_edge):
    '''Calculate the total mismatch between two binning schemes.
    Pass bin content and bin edges of the two histograms for comparison'''
    metric = 0
    for i in range(len(be_test)-1):
        low_edge_test         = be_test[i]
        hi_edge_test          = be_test[i+1]
        ni1, low_nom_low_test = find_le(be_nominal, low_edge_test)
        hi_nom_low_test       = find_gt(be_nominal, low_edge_test)[1]
        ni2, low_nom_hi_test  = find_lt(be_nominal, hi_edge_test)
        hi_nom_hi_test        = find_ge(be_nominal, hi_edge_test)[1]
        bc_nom_1              = bc_nominal[ni1]
        try:
            bc_nom_2 = bc_nominal[ni2]
        except:
            bc_nom_2 = -1

        if low_edge_test == low_nom_low_test and hi_edge_test == hi_nom_hi_test:
            # low and high edges for nominal and test match, ignore them and go to next bin
            continue
        elif low_nom_low_test == low_nom_hi_test and hi_nom_low_test == hi_nom_hi_test:
            # test bin completely contained in nominal bin (easy case)
            if low_edge_test == main_edge:
                metric += 2*(bc_test[i]-bc_nom_1)
            else:
                metric += abs(bc_nom_1 - bc_test[i])
        else:
            # test bin overlaps two bg bins (harder case)
            width_test = hi_edge_test - low_edge_test
            low_width_test = hi_nom_low_test - low_edge_test
            hi_width_test = hi_edge_test - hi_nom_low_test
            if low_edge_test == main_edge:
                metric += 2*(bc_test[i]-bc_nom_1)*(low_width_test/width_test)
                metric += 2*(bc_test[i]-bc_nom_2)*(hi_width_test/width_test)
            else:
                metric += abs(bc_nom_1-bc_test[i])*(low_width_test/width_test)
                metric += abs(bc_nom_2-bc_test[i])*(hi_width_test/width_test)
    return metric

def get_mismatch_metric_v3(bc_nominal, be_nominal, bc_test, be_test, main_edge):
    '''Calculate the total mismatch between two binning schemes.
    Pass bin content and bin edges of the two histograms for comparison'''
    metric = 0
    for i in range(len(be_test)-1):
        low_edge_test         = be_test[i]
        hi_edge_test          = be_test[i+1]
        ni1, low_nom_low_test = find_le(be_nominal, low_edge_test)
        hi_nom_low_test       = find_gt(be_nominal, low_edge_test)[1]
        ni2, low_nom_hi_test  = find_lt(be_nominal, hi_edge_test)
        hi_nom_hi_test        = find_ge(be_nominal, hi_edge_test)[1]
        bc_nom_1              = bc_nominal[ni1]
        try:
            bc_nom_2 = bc_nominal[ni2]
        except:
            bc_nom_2 = -1

        if low_edge_test == low_nom_low_test and hi_edge_test == hi_nom_hi_test:
            # low and high edges for nominal and test match, ignore them and go to next bin
            continue
        elif low_nom_low_test == low_nom_hi_test and hi_nom_low_test == hi_nom_hi_test:
            # test bin completely contained in nominal bin (easy case)
            if low_edge_test == main_edge:
                metric += 2*(bc_test[i]-bc_nom_1)
        else:
            # test bin overlaps two bg bins (harder case)
            width_test = hi_edge_test - low_edge_test
            low_width_test = hi_nom_low_test - low_edge_test
            hi_width_test = hi_edge_test - hi_nom_low_test
            if low_edge_test == main_edge:
                metric += 2*(bc_test[i]-bc_nom_1)*(low_width_test/width_test)
                metric += 2*(bc_test[i]-bc_nom_2)*(hi_width_test/width_test)
    return metric


def calc_local_pvalue(N_bg, var_bg, N_sig, var_sig, ntoys=1e7):
    '''Not an accurate estimate for large sigma'''
    print ''
    print 'Calculating local p-value and significance based on {0} toys'.format(ntoys)
    print 'N_bg = {0}, sigma_bg = {1}, N_signal = {2}'.format(N_bg,var_bg,N_sig)
    toys    = np.random.normal(N_bg, var_bg, int(ntoys))
    #print toys
    pvars   = np.random.poisson(toys)
    pval    = pvars[pvars > N_bg + N_sig].size/ntoys
    print 'local p-value = {0}'.format(pval)
    print 'local significance = {0:.2f}'.format(np.abs(norm.ppf(1-pval)))

def generate_initial_params(hgg_bg, hgg_signal, n_sigma):
    '''Input bg and signal dataframes, and a sigma value for signal injection.
    Output parameters for the pdfs that describe those distributions.'''
    # grab a handful of bg events, and an ~X sigma number of signal events
    hgg_bg_selection     = hgg_bg[(hgg_bg.Mgg > 100) & (hgg_bg.Mgg < 180)][0:10000].Mgg
    n_bg_under_sig       = hgg_bg_selection[(118 < hgg_bg_selection) & (hgg_bg_selection < 133)].size
    n_sig                = int(n_sigma*np.sqrt(n_bg_under_sig))
    hgg_signal_selection = hgg_signal[(hgg_signal.Mgg >= 118) & (hgg_signal.Mgg <= 133)][0:n_sig].Mgg
    data_bg              = hgg_bg_selection.values
    data_sig             = hgg_signal_selection.values

    # fit to the data distributions
    bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
    bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])
    bg_fitter = ff.NLLFitter(bg_model, data_bg)
    bg_result = bg_fitter.fit([0.0, 0.0, 0.0])

    sig_model = ff.Model(sig_pdf, ['mu', 'sigma'])
    sig_model.set_bounds([(110, 130), (1, 5)])
    sig_fitter = ff.NLLFitter(sig_model, data_sig)
    sig_result = sig_fitter.fit([120.0, 2])

    n_bg = len(data_bg)

    be_bg = bayesian_blocks(data_bg, p0=0.02)
    be_sig = bayesian_blocks(data_sig, p0=0.02)

    return bg_result, sig_result, n_bg, n_sig, be_bg, be_sig

def generate_toy_data(bg_params, sig_params, n_bg, n_sig, seed=None):
    '''use bg and signal params to generated simulated data'''
    # bg dist
    bg_pdf_ROOT = functools.partial(bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 100, 180, 3)
    tf1_bg_pdf.SetParameters(*bg_params)
    # signal dist
    sig_pdf_ROOT = functools.partial(sig_pdf, doROOT=True)
    tf1_sig_pdf = TF1("tf1_sig_pdf", sig_pdf_ROOT, 100, 180, 2)
    tf1_sig_pdf.SetParameters(*sig_params)
    mc_bg = [tf1_bg_pdf.GetRandom() for i in xrange(n_bg)]
    mc_sig = [tf1_sig_pdf.GetRandom() for i in xrange(n_sig)]
    return mc_bg, mc_sig

def generate_q0_via_nll_unbinned(data, bg_params=None, sig_params=None):
    '''Perform two nll fits to data, one for bg+signal, one for bg-only.
    Use these values to create the q0 statistic.'''

    bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
    bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])

    bg_sig_model = ff.Model(bg_sig_pdf, ['C', 'mu', 'sigma', 'a1', 'a2', 'a3'])
    bg_sig_model.set_bounds([(0, 1), (111, 179), (0.01, 10), (-1., 1.), (-1., 1.), (-1., 1.)])

    if bg_params:
        bg_nll = bg_model.nll(np.asarray(data),bg_params)
    else:
        mc_bg_only_fitter = ff.NLLFitter(bg_model, np.asarray(data),verbose=False)
        mc_bg_only_result = mc_bg_only_fitter.fit([ 0.0, 0.0, 0.0], calculate_corr = False)
        bg_nll = mc_bg_only_result.fun
    if sig_params:
        bg_sig_nll = bg_sig_model.nll(np.asarray(data),[0.02306,125.772,2.775,-0.957,0.399,-0.126])
    else:
        mc_bg_sig_fitter = ff.NLLFitter(bg_sig_model, np.asarray(data),verbose=False)
        mc_bg_sig_result = mc_bg_sig_fitter.fit([0.01, 125.0, 2, 0.0, 0.0, 0.0], calculate_corr = False)
        bg_sig_nll = mc_bg_sig_result.fun
    q0 = 2*max(bg_nll-bg_sig_nll,0)
    return q0

def generate_q0_via_bins(data, bin_edges, true_bg_bc, true_sig_bc):
    '''Generate likelihood ratios based on poisson distributions for each bin
    in binned data.  True values for bg and bg+signal are determined from integration of
    underlying pdfs used to generate toys.
    Use these values to create the q0 statistic.'''

    bc, bin_edges = np.histogram(data, bin_edges, range=(100,180))
    l_bg  = 1
    l_sig = 1
    for i in range(len(bin_edges)-1):
        l_bg  *= poisson.pmf(bc[i], true_bg_bc[i])
        l_sig *= poisson.pmf(bc[i], true_bg_bc[i]+true_sig_bc[i])

    q0 = -2*(np.log(l_bg)-np.log(l_sig))
    return q0

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

    true_bg_bc_1GeV  = []
    true_sig_bc_1GeV = []
    for i in range(len(be_1GeV)-1):
	true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),be_1GeV[i],be_1GeV[i+1])
	true_bg_bc_1GeV.append(true_bg*n_bg)
	true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),be_1GeV[i],be_1GeV[i+1])
	true_sig_bc_1GeV.append(true_sig*n_sig)
    true_bg_bc_2GeV  = []
    true_sig_bc_2GeV = []
    for i in range(len(be_2GeV)-1):
	true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),be_2GeV[i],be_2GeV[i+1])
	true_bg_bc_2GeV.append(true_bg*n_bg)
	true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),be_2GeV[i],be_2GeV[i+1])
	true_sig_bc_2GeV.append(true_sig*n_sig)
    true_bg_bc_5GeV  = []
    true_sig_bc_5GeV = []
    for i in range(len(be_5GeV)-1):
	true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),be_5GeV[i],be_5GeV[i+1])
	true_bg_bc_5GeV.append(true_bg*n_bg)
	true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),be_5GeV[i],be_5GeV[i+1])
	true_sig_bc_5GeV.append(true_sig*n_sig)
    true_bg_bc_10GeV  = []
    true_sig_bc_10GeV = []
    for i in range(len(be_10GeV)-1):
	true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),be_10GeV[i],be_10GeV[i+1])
	true_bg_bc_10GeV.append(true_bg*n_bg)
	true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),be_10GeV[i],be_10GeV[i+1])
	true_sig_bc_10GeV.append(true_sig*n_sig)

    signif_nll_fit_hist = []
    signif_nll_true_hist = []
    signif_bb_true_hist = []
    signif_1GeV_true_hist = []
    signif_2GeV_true_hist = []
    signif_5GeV_true_hist = []
    signif_10GeV_true_hist = []

    # Do a bunch of toys
    gRandom.SetSeed(10)

    for i in range(500):
        if i%10==0: print 'doing fit #',i
        mc_bg, mc_sig = generate_toy_data(bg_result.x, sig_result.x, n_bg, n_sig)
        mc_bg_sig = mc_bg+mc_sig

        q0_nll_fit = generate_q0_via_nll_unbinned(mc_bg_sig)
        signif_nll_fit_hist.append(np.sqrt(q0_nll_fit))

        q0_nll_true = generate_q0_via_nll_unbinned(mc_bg_sig,
                bg_params = [-0.957,0.399,-0.126],
                sig_params = [0.02306,125.772,2.775,-0.957,0.399,-0.126])
        signif_nll_true_hist.append(np.sqrt(q0_nll_true))

        q0_bb_true = generate_q0_via_bins(mc_bg_sig, be_hybrid, true_bg_bc, true_sig_bc)
        signif_bb_true_hist.append(np.sqrt(q0_bb_true))

        q0_1GeV_true = generate_q0_via_bins(mc_bg_sig, be_1GeV, true_bg_bc_1GeV, true_sig_bc_1GeV)
        signif_1GeV_true_hist.append(np.sqrt(q0_1GeV_true))

        q0_2GeV_true = generate_q0_via_bins(mc_bg_sig, be_2GeV, true_bg_bc_2GeV, true_sig_bc_2GeV)
        signif_2GeV_true_hist.append(np.sqrt(q0_2GeV_true))

        q0_5GeV_true = generate_q0_via_bins(mc_bg_sig, be_5GeV, true_bg_bc_5GeV, true_sig_bc_5GeV)
        signif_5GeV_true_hist.append(np.sqrt(q0_5GeV_true))

        q0_10GeV_true = generate_q0_via_bins(mc_bg_sig, be_10GeV, true_bg_bc_10GeV, true_sig_bc_10GeV)
        signif_10GeV_true_hist.append(np.sqrt(q0_10GeV_true))

