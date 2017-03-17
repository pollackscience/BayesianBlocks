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
from lmfit import Model
from bb.tools.bayesian_blocks_modified import bayesian_blocks
import nllfitter.future_fitter as ff
from ROOT import gRandom
from ROOT import TF1
from tqdm import tqdm


def lm_binned_wrapper(mu_bg, mu_sig):
    def lm_binned(ix, A, ntot):
        proper_means = ((mu_bg+A*mu_sig)/np.sum(mu_bg+A*mu_sig))*ntot
        return proper_means[ix]
    return lm_binned


def bg_pdf_simp(x, a, doROOT=False):
    '''BG parameterization from BH analysis:
    f(x)=(A(1+x)^alpha)/(x^(beta+gamma*ln(x)))
    a[0]: A     normalization parameter
    a[1]: alpha free parameter
    x:    ST    independent var
    '''
    if doROOT:
        x = x[0]  # This is a ROOT-compatible hack
    # print a[0]*((1+x)**a[1])/(x**(a[2]+a[3]*np.log(x)))
    return (1./a[1])*np.exp(-(x-a[0])/a[1])


# def bg_pdf_helper(x, a0, a2):
def bg_pdf_helper(x, a0, a1, a2):
    return ((1+x)**a0)/(x**(a1+a2*np.log(x)))
    # return ((1+x)**a0)/(x**(a2*np.log(x)))


def bg_pdf(x, a, xlow=2800, xhigh=13000, doROOT=False):
    '''BG parameterization from BH analysis:
    f(x)=((1+x)^alpha)/(x^(beta+gamma*ln(x)))
    a[0]: alpha free parameter
    a[1]: beta  free parameter
    a[2]: gamma free parameter
    x:    ST    independent var
    '''
    if doROOT:
        x = x[0]  # This is a ROOT-compatible hack

    func = functools.partial(bg_pdf_helper, a0=a[0], a1=a[1], a2=a[2])
    # func = functools.partial(bg_pdf_helper, a0=a[0], a2=a[1])
    return func(x)/integrate.quad(func, 2800, 13000)[0]


def sig_pdf(x, a, doROOT=False):
    '''simple gaussian pdf.
    a[0]: mean
    a[1]: sigma
    '''
    if doROOT:
        x = x[0]  # This is a ROOT-compatible hack
    return (1.0/(a[1]*np.sqrt(2*np.pi)))*np.exp(-(x-a[0])**2/(2*a[1]**2))


def bg_sig_pdf(x, a, xlow=2800, xhigh=13000, doROOT=False):
    '''BH bg pdf and gaussian signal pdf, with a relative normalization factor.
    a[0]: normalization and signal strength parameter
    a[1]: signal mean
    a[2]: signal sigma
    a[3]: alpha   free parameter
    a[4]: beta    free parameter
    a[5]: gamma   free parameter
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
        return i-1, a[i-1]
    raise ValueError


def find_lt(a, x):
    '''Find rightmost value less than x'''
    i = bisect.bisect_left(a, x)
    if i:
        return i-1, a[i-1]
    raise ValueError


def find_ge(a, x):
    '''Find leftmost item greater than or equal to x'''
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i, a[i]
    raise ValueError


def find_gt(a, x):
    '''Find leftmost value greater than x'''
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i, a[i]
    raise ValueError


def calc_local_pvalue(N_bg, var_bg, N_sig, var_sig, ntoys=1e7):
    '''Not an accurate estimate for large sigma'''
    print ''
    print 'Calculating local p-value and significance based on {0} toys'.format(ntoys)
    print 'N_bg = {0}, sigma_bg = {1}, N_signal = {2}'.format(N_bg, var_bg, N_sig)
    toys    = np.random.normal(N_bg, var_bg, int(ntoys))
    pvars   = np.random.poisson(toys)
    pval    = pvars[pvars > N_bg + N_sig].size/ntoys
    print 'local p-value = {0}'.format(pval)
    print 'local significance = {0:.2f}'.format(np.abs(norm.ppf(1-pval)))


def generate_initial_params(data_bg_mul2, data_bg_mul8):

    # fit to the data distributions
    bg_model = ff.Model(bg_pdf, ['alpha', 'beta', 'gamma'])
    bg_model.set_bounds([(-200, 200), (-100, 100), (-100,100)])
    bg_fitter = ff.NLLFitter(bg_model, data_bg_mul2)
    bg_result = bg_fitter.fit([-1.80808e+01, -8.21174e-02, 8.06289e-01])

    #sig_model = ff.Model(sig_pdf, ['mu', 'sigma'])
    #sig_model.set_bounds([(110, 130), (1, 5)])
    #sig_fitter = ff.NLLFitter(sig_model, data_sig)
    #sig_result = sig_fitter.fit([120.0, 2])

    n_bg = len(data_bg_mul8)

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
    if bg_params:
        bg_model.set_bounds([(bg_params[0], bg_params[0]), (bg_params[1], bg_params[1]),
                             (bg_params[2], bg_params[2])])
    else:
        bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])

    bg_sig_model = ff.Model(bg_sig_pdf, ['C', 'mu', 'sigma', 'a1', 'a2', 'a3'])
    if sig_params:
        if len(sig_params) == 5:
            bg_sig_model.set_bounds([(0, 1), (sig_params[0], sig_params[0]),
                                     (sig_params[1], sig_params[1]),
                                     (sig_params[2], sig_params[2]),
                                     (sig_params[3], sig_params[3]),
                                     (sig_params[4], sig_params[4])])
        else:
            bg_sig_model.set_bounds([(sig_params[0], sig_params[0]),
                                     (sig_params[1], sig_params[1]),
                                     (sig_params[2], sig_params[2]),
                                     (sig_params[3], sig_params[3]),
                                     (sig_params[4], sig_params[4]),
                                     (sig_params[5], sig_params[5])])
    else:
        bg_sig_model.set_bounds([(0, 1), (120, 130), (1, 4), (-1., 1.), (-1., 1.), (-1., 1.)])

    mc_bg_only_fitter = ff.NLLFitter(bg_model, np.asarray(data), verbose=False)
    if bg_params:
        mc_bg_only_result = mc_bg_only_fitter.fit([bg_params[0], bg_params[1], bg_params[2]],
                                                  calculate_corr=False)
    else:
        mc_bg_only_result = mc_bg_only_fitter.fit([-0.963, 0.366, -0.091], calculate_corr=False)
    bg_nll = mc_bg_only_result.fun

    mc_bg_sig_fitter = ff.NLLFitter(bg_sig_model, np.asarray(data), verbose=False)
    if sig_params:
        if len(sig_params) == 5:
            mc_bg_sig_result = mc_bg_sig_fitter.fit([0.01, sig_params[0], sig_params[1],
                                                     sig_params[2], sig_params[3], sig_params[4]],
                                                    calculate_corr=False)
        else:
            mc_bg_sig_result = mc_bg_sig_fitter.fit([sig_params[0], sig_params[1], sig_params[2],
                                                     sig_params[3], sig_params[4], sig_params[5]],
                                                    calculate_corr=False)
    else:
        mc_bg_sig_result = mc_bg_sig_fitter.fit([0.01, 125.77, 2.775, -0.957, 0.399, -0.126],
                                                calculate_corr=False)
    bg_sig_nll = mc_bg_sig_result.fun
    q0 = 2*max(bg_nll-bg_sig_nll, 0)
    return q0


def generate_q0_via_nll_unbinned_constrained(bg, data):
    '''Perform two nll fits to data, one for bg+signal, one for bg-only.
    Use these values to create the q0 statistic.'''

    data = np.asarray(data)
    bg = np.asarray(bg)
    bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
    bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])

    mc_bg_only_fitter = ff.NLLFitter(bg_model, bg, verbose=False)
    mc_bg_only_result = mc_bg_only_fitter.fit([-0.963, 0.366, -0.091], calculate_corr=False)
    bg_ps = mc_bg_only_result.x
    bg_nll = bg_model.nll(data, bg_ps)

    bg_sig_model = ff.Model(bg_sig_pdf, ['C', 'mu', 'sigma', 'a1', 'a2', 'a3'])
    bg_sig_model.set_bounds([(0, 1), (125.77, 125.77), (2.775, 2.775), (bg_ps[0], bg_ps[0]),
                             (bg_ps[1], bg_ps[1]), (bg_ps[2], bg_ps[2])])

    mc_bg_sig_fitter = ff.NLLFitter(bg_sig_model, np.asarray(data), verbose=False)
    mc_bg_sig_result = mc_bg_sig_fitter.fit([0.01, 125.77, 2.775, bg_ps[0], bg_ps[1], bg_ps[2]],
                                            calculate_corr=False)
    bg_sig_nll = mc_bg_sig_result.fun
    q0 = 2*max(bg_nll-bg_sig_nll, 0)

    return q0


def generate_q0_via_bins(data, bin_edges, true_bg_bc, true_sig_bc):
    '''Generate likelihood ratios based on poisson distributions for each bin
    in binned data.  True values for bg and bg+signal are determined from integration of
    underlying pdfs used to generate toys.
    Use these values to create the q0 statistic.'''

    bc, bin_edges = np.histogram(data, bin_edges, range=(100, 180))
    l_bg  = 1
    l_sig = 1
    for i in range(len(bin_edges)-1):
        l_bg  *= poisson.pmf(bc[i], true_bg_bc[i])
        l_sig *= poisson.pmf(bc[i], true_bg_bc[i]+true_sig_bc[i])

    q0 = -2*(np.log(l_bg)-np.log(l_sig))
    return q0


def generate_q0_via_shape_fit(data, bin_edges, binned_model, binned_params):
    '''Generate likelihood ratios based on a template fit to the data.
    Shape values for bg and signal are determined from integration of
    underlying pdfs used to generate toys.
    Use these values to create the q0 statistic.'''

    bc, bin_edges = np.histogram(data, bin_edges, range=(100, 180))
    ibc = np.asarray(range(len(bc)))
    result = binned_model.fit(bc, ix=ibc, params=binned_params)
    nll_bg = -np.sum(np.log(poisson.pmf(bc, result.eval(A=0))))
    nll_sig = -np.sum(np.log(poisson.pmf(bc, result.best_fit)))

    q0 = 2*(nll_bg-nll_sig)
    return q0


if __name__ == "__main__":
    plt.close('all')
    bb_dir  = os.path.join(os.path.dirname(__file__), '../..')
    df_data_mul2 = pkl.load(open(bb_dir+'/files/BH/BH_test_data.p','rb'))
    data_bg_mul2 = df_data_mul2[df_data_mul2.ST_mul2_BB>=2800].ST_mul2_BB.values

    df_data_mul8 = pkl.load(open(bb_dir+'/files/BH/BH_paper_data.p','rb'))
    data_bg_mul8 = df_data_mul8[df_data_mul8.ST_mul8_BB>=2800].ST_mul8_BB.values

    bg_result, sig_result, n_bg, n_sig, be_bg, be_sig = generate_initial_params(
        hgg_bg, hgg_signal, 5)

    be_100GeV = np.linspace(2800, 13000, 103)

    true_bg_bc  = []
    true_sig_bc = []
    for i in range(len(be_hybrid)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_hybrid[i], be_hybrid[i+1])
        true_bg_bc.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),
                                      be_hybrid[i], be_hybrid[i+1])
        true_sig_bc.append(true_sig*n_sig)

    lm_binned = lm_binned_wrapper(np.asarray(true_bg_bc), np.asarray(true_sig_bc))
    binned_model = Model(lm_binned)
    binned_params = binned_model.make_params()
    binned_params['ntot'].value   = n_bg+n_sig
    binned_params['ntot'].vary    = False
    binned_params['A'].value      = 0.1
    binned_params['A'].min        = 0
    binned_params['A'].max        = 1000

    true_bg_bc_1GeV  = []
    true_sig_bc_1GeV = []
    for i in range(len(be_1GeV)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_1GeV[i], be_1GeV[i+1])
        true_bg_bc_1GeV.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),
                                      be_1GeV[i], be_1GeV[i+1])
        true_sig_bc_1GeV.append(true_sig*n_sig)

    true_bg_bc_2GeV  = []
    true_sig_bc_2GeV = []
    for i in range(len(be_2GeV)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_2GeV[i], be_2GeV[i+1])
        true_bg_bc_2GeV.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),
                                      be_2GeV[i], be_2GeV[i+1])
        true_sig_bc_2GeV.append(true_sig*n_sig)

    true_bg_bc_5GeV  = []
    true_sig_bc_5GeV = []
    for i in range(len(be_5GeV)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_5GeV[i], be_5GeV[i+1])
        true_bg_bc_5GeV.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),
                                      be_5GeV[i], be_5GeV[i+1])
        true_sig_bc_5GeV.append(true_sig*n_sig)

    true_bg_bc_10GeV  = []
    true_sig_bc_10GeV = []
    for i in range(len(be_10GeV)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_10GeV[i], be_10GeV[i+1])
        true_bg_bc_10GeV.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),
                                      be_10GeV[i], be_10GeV[i+1])
        true_sig_bc_10GeV.append(true_sig*n_sig)

    signif_nll_fit_hist = []
    signif_nll_constrained_hist = []
    signif_nll_true_hist = []
    signif_nll_true_fit_hist = []
    signif_bb_true_hist = []
    signif_bb_shape_hist = []
    signif_1GeV_true_hist = []
    signif_2GeV_true_hist = []
    signif_5GeV_true_hist = []
    signif_10GeV_true_hist = []

    # Do a bunch of toys
    gRandom.SetSeed(20)

    for i in tqdm(range(500)):
        mc_bg, mc_sig = generate_toy_data(bg_result.x, sig_result.x, n_bg, n_sig)
        mc_bg_sig = mc_bg+mc_sig

        q0_nll_fit = generate_q0_via_nll_unbinned(mc_bg_sig)
        signif_nll_fit_hist.append(np.sqrt(q0_nll_fit))

        q0_nll_constrained = generate_q0_via_nll_unbinned_constrained(mc_bg, mc_bg_sig)
        signif_nll_constrained_hist.append(np.sqrt(q0_nll_constrained))

        q0_nll_true = generate_q0_via_nll_unbinned(mc_bg_sig, bg_params=[-0.957, 0.399, -0.126],
                                                   sig_params=[0.02306, 125.772, 2.775, -0.957,
                                                               0.399, -0.126])
        signif_nll_true_hist.append(np.sqrt(q0_nll_true))

        q0_nll_true_fit = generate_q0_via_nll_unbinned(mc_bg_sig, bg_params=[-0.957, 0.399, -0.126],
                                                       sig_params=[125.772, 2.775, -0.957, 0.399,
                                                                   -0.126])
        signif_nll_true_fit_hist.append(np.sqrt(q0_nll_true_fit))

        q0_bb_true = generate_q0_via_bins(mc_bg_sig, be_hybrid, true_bg_bc, true_sig_bc)
        signif_bb_true_hist.append(np.sqrt(q0_bb_true))

        q0_bb_shape = generate_q0_via_shape_fit(mc_bg_sig, be_hybrid, binned_model, binned_params)
        signif_bb_shape_hist.append(np.sqrt(q0_bb_shape))

        q0_1GeV_true = generate_q0_via_bins(mc_bg_sig, be_1GeV, true_bg_bc_1GeV, true_sig_bc_1GeV)
        signif_1GeV_true_hist.append(np.sqrt(q0_1GeV_true))

        q0_2GeV_true = generate_q0_via_bins(mc_bg_sig, be_2GeV, true_bg_bc_2GeV, true_sig_bc_2GeV)
        signif_2GeV_true_hist.append(np.sqrt(q0_2GeV_true))

        q0_5GeV_true = generate_q0_via_bins(mc_bg_sig, be_5GeV, true_bg_bc_5GeV, true_sig_bc_5GeV)
        signif_5GeV_true_hist.append(np.sqrt(q0_5GeV_true))

        q0_10GeV_true = generate_q0_via_bins(mc_bg_sig, be_10GeV,
                                             true_bg_bc_10GeV, true_sig_bc_10GeV)
        signif_10GeV_true_hist.append(np.sqrt(q0_10GeV_true))
