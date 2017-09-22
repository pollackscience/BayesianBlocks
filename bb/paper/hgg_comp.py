#! /usr/bin/env python

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import functools
import bisect
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy.stats import poisson
import pandas as pd
from matplotlib import pyplot as plt
# from lmfit import Model
from lmfit import Parameters
from bb.tools.bayesian_blocks_modified import bayesian_blocks
# import nllfitter.future_fitter as ff
from nllfit import NLLFitter, Model
from ROOT import gRandom
from ROOT import TF1
from tqdm import tqdm_notebook, tnrange
from six.moves import range


def lm_binned_wrapper(mu_bg, mu_sig):
    def lm_binned(ix, A, ntot):
        proper_means = ((mu_bg+A*mu_sig)/np.sum(mu_bg+A*mu_sig))*ntot
        return proper_means[ix]
    return lm_binned


def template_pdf_wrapper(mu_bg, mu_sig, cnc=False):
    '''Wrapper function to produce a pmf of poisson variables based on a bg+signal mixture model.
    Input means are automatically normalized.

    mu_bg:  Collection of expected background yields for all bins.
    mu_sig: Collection of expected signal yields for all bins.

    mu_bg and mu_sig must be the same length.'''

    if len(mu_bg) != len(mu_sig):
        raise Exception('mu_bg must be the same length as mu_sig!')

    mu_bg = np.asarray(mu_bg)
    mu_sig = np.asarray(mu_sig)
    if not cnc:
        mu_bg = mu_bg/float(np.sum(mu_bg))
        mu_sig = mu_sig/float(np.sum(mu_sig))

    def template_pdf(x, a):
        '''Binned template pdf for simple mixture model.

        x:      bin content of data histogram
        a[0]:   signal amplitude
        a[1]:   total number of events (should be fixed).

        x must be the same length as mu_bg (and mu_sig).'''

        if len(x) != len(mu_bg):
            raise Exception('x must be the same length as mu_bg!')

        means = ((1-a[0])*mu_bg+a[0]*mu_sig)*a[1]
        pdf = poisson.pmf(x, means)
        return pdf

    return template_pdf


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
    print('')
    print('Calculating local p-value and significance based on {0} toys'.format(ntoys))
    print('N_bg = {0}, sigma_bg = {1}, N_signal = {2}'.format(N_bg, var_bg, N_sig))
    toys    = np.random.normal(N_bg, var_bg, int(ntoys))
    pvars   = np.random.poisson(toys)
    pval    = pvars[pvars > N_bg + N_sig].size/ntoys
    print('local p-value = {0}'.format(pval))
    print('local significance = {0:.2f}'.format(np.abs(norm.ppf(1-pval))))


def generate_initial_params(hgg_bg, hgg_signal, n_sigma):
    '''Input bg and signal dataframes, and a sigma value for signal injection.
    Output parameters for the pdfs that describe those distributions.'''
    # grab a handful of bg events, and an ~X sigma number of signal events
    hgg_bg_selection     = hgg_bg[(hgg_bg.Mgg > 100) & (hgg_bg.Mgg < 180)][0:10000].Mgg
    n_bg_under_sig       = hgg_bg_selection[(118 < hgg_bg_selection) &
                                            (hgg_bg_selection < 133)].size
    n_sig                = int(n_sigma*np.sqrt(n_bg_under_sig))
    hgg_signal_selection = hgg_signal[(hgg_signal.Mgg >= 118) &
                                      (hgg_signal.Mgg <= 133)][0:n_sig].Mgg
    data_bg              = hgg_bg_selection.values
    data_sig             = hgg_signal_selection.values

    # fit to the data distributions
    bg_params = Parameters()
    bg_params.add_many(
        ('a1', 0., True, -1, 1, None, None),
        ('a2', 0., True, -1, 1, None, None),
        ('a3', 0., True, -1, 1, None, None)
    )

    bg_model = Model(bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model)
    bg_result = bg_fitter.fit(data_bg, calculate_corr=False)

    # bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
    # bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])

    # bg_fitter = ff.NLLFitter(bg_model, data_bg)
    # bg_result = bg_fitter.fit([0.0, 0.0, 0.0])

    # sig_model = ff.Model(sig_pdf, ['mu', 'sigma'])
    # sig_model.set_bounds([(110, 130), (1, 5)])
    # sig_fitter = ff.NLLFitter(sig_model, data_sig)
    # sig_result = sig_fitter.fit([120.0, 2])

    sig_params = Parameters()
    sig_params.add_many(
        ('mu'    , 125 , True , 110 , 130  , None, None),
		('sigma' , 1 , True , 1 , 5   , None, None),
    )
    sig_model  = Model(sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data_sig)

    n_bg = len(data_bg)

    be_bg = bayesian_blocks(data_bg, p0=0.02)
    be_sig = bayesian_blocks(data_sig, p0=0.02)

    return bg_result, sig_result, n_bg, n_sig, be_bg, be_sig


def generate_toy_data(bg_pdf_ROOT, sig_pdf_ROOT, n_bg, n_sig, seed=None):
    '''use bg and signal params to generated simulated data'''
    # bg dist
    mc_bg = [bg_pdf_ROOT.GetRandom() for i in range(n_bg)]
    mc_sig = [sig_pdf_ROOT.GetRandom() for i in range(n_sig)]
    return mc_bg, mc_sig


def generate_q0_via_nll_unbinned(data, bg_params=None, sig_params=None):
    '''Perform two nll fits to data, one for bg+signal, one for bg-only.
    Use these values to create the q0 statistic.'''

    if not bg_params:
        _bg_params = Parameters()
        _bg_params.add_many(
            ('a1', 0., True, -1, 1, None, None),
            ('a2', 0., True, -1, 1, None, None),
            ('a3', 0., True, -1, 1, None, None)
        )
    else:
        _bg_params = Parameters()
        _bg_params.add_many(
            ('a1', bg_params[0], False, -1, 1, None, None),
            ('a2', bg_params[1], False, -1, 1, None, None),
            ('a3', bg_params[2], False, -1, 1, None, None)
        )

    bg_model = Model(bg_pdf, _bg_params)

    if not sig_params:
        _sig_params = Parameters()
        _sig_params.add_many(
            ('C'     , 0.1 , True , 0   , 1   , None , None) ,
            ('mu'    , 125 , True , 120 , 130 , None , None) ,
            ('sigma' , 2   , True , 1   , 4   , None , None) ,
            ('a1'    , 0.  , True , -1  , 1   , None , None) ,
            ('a2'    , 0.  , True , -1  , 1   , None , None) ,
            ('a3'    , 0.  , True , -1  , 1   , None , None)
        )
    else:
        if len(sig_params) == 5:
            _sig_params = Parameters()
            _sig_params.add_many(
                ('C'     , 0.1 , True , 0   , 1   , None , None) ,
                ('mu'    , sig_params[0], False, 120 , 130 , None , None) ,
                ('sigma' , sig_params[1], False , 1   , 4   , None , None) ,
                ('a1'    , sig_params[2], False , -1  , 1   , None , None) ,
                ('a2'    , sig_params[3], False , -1  , 1   , None , None) ,
                ('a3'    , sig_params[4], False , -1  , 1   , None , None)
            )
        else:
            _sig_params = Parameters()
            _sig_params.add_many(
                ('C'     , sig_params[0] , False, 0   , 1   , None , None) ,
                ('mu'    , sig_params[1], False, 120 , 130 , None , None) ,
                ('sigma' , sig_params[2], False , 1   , 4   , None , None) ,
                ('a1'    , sig_params[3], False , -1  , 1   , None , None) ,
                ('a2'    , sig_params[4], False , -1  , 1   , None , None) ,
                ('a3'    , sig_params[5], False , -1  , 1   , None , None)
            )

    bg_sig_model = Model(bg_sig_pdf, _sig_params)

    mc_bg_only_fitter = NLLFitter(bg_model)
    mc_bg_only_result = mc_bg_only_fitter.fit(np.asarray(data), calculate_corr=False, verbose=False)
    bg_nll = mc_bg_only_result.fun

    mc_bg_sig_fitter = NLLFitter(bg_sig_model)
    mc_bg_sig_result = mc_bg_sig_fitter.fit(np.asarray(data), calculate_corr=False, verbose=False)
    bg_sig_nll = mc_bg_sig_result.fun
    q0 = 2*max(bg_nll-bg_sig_nll, 0)
    return q0


def generate_q0_via_nll_unbinned_constrained(bg, data, bg_params):
    '''Perform two nll fits to data, one for bg+signal, one for bg-only.
    Use these values to create the q0 statistic.'''

    data = np.asarray(data)
    bg = np.asarray(bg)
    _bg_params = Parameters()
    _bg_params.add_many(
        ('a1', bg_params[0], False, -1, 1, None, None),
        ('a2', bg_params[1], False, -1, 1, None, None),
        ('a3', bg_params[2], False, -1, 1, None, None)
    )

    bg_model = Model(bg_pdf, _bg_params)
    mc_bg_only_fitter = NLLFitter(bg_model)
    mc_bg_only_fitter.fit(bg, calculate_corr=False, verbose=False)

    bg_nll = bg_model.calc_nll(None, data)

    _sig_params = Parameters()
    _sig_params.add_many(
        ('C'     , 0.1                    , True  , 0   , 1   , None , None) ,
        ('mu'    , 125.77                 , False , 120 , 130 , None , None) ,
        ('sigma' , 2.775                  , False , 1   , 4   , None , None) ,
        ('a1'    , _bg_params['a1'].value , False , -1  , 1   , None , None) ,
        ('a2'    , _bg_params['a2'].value , False , -1  , 1   , None , None) ,
        ('a3'    , _bg_params['a3'].value , False , -1  , 1   , None , None)
    )

    bg_sig_model = Model(bg_sig_pdf, _sig_params)

    mc_bg_sig_fitter = NLLFitter(bg_sig_model)
    mc_bg_sig_result = mc_bg_sig_fitter.fit(data, calculate_corr=False, verbose=False)
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


def generate_q0_via_shape_fit(data, bin_edges, template_params, template_pdf):
    '''Generate likelihood ratios based on a template fit to the data.
    Shape values for bg and signal are determined from integration of
    underlying pdfs used to generate toys.
    Use these values to create the q0 statistic.'''

    n_tot = len(data)
    bc, bin_edges = np.histogram(data, bin_edges, range=(100, 180))

    _template_params = template_params.copy()
    _template_params['n_tot'].value = n_tot
    template_model = Model(template_pdf, _template_params)
    template_fitter = NLLFitter(template_model)
    mle_res = template_fitter.fit(bc, calculate_corr=False, verbose=False)
    nll_sig = mle_res.fun

    _template_params = template_params.copy()
    _template_params['n_tot'].value = n_tot
    _template_params['A'].value = 0
    _template_params['A'].vary = False
    template_model = Model(template_pdf, _template_params)
    template_fitter = NLLFitter(template_model)
    bg_res = template_fitter.fit(bc, calculate_corr=False, verbose=False)
    nll_bg = bg_res.fun

    q0 = 2*(nll_bg-nll_sig)
    return q0


if __name__ == "__main__":
    plt.close('all')
    current_dir = os.path.dirname(__file__)
    bb_dir      = os.path.join(current_dir, '../..')
    hgg_bg      = pd.read_pickle(bb_dir+'/files/hgg_bg.p')
    hgg_signal  = pd.read_pickle(bb_dir+'/files/hgg_signal.p')

    bg_result, sig_result, n_bg, n_sig, be_bg, be_sig = generate_initial_params(
        hgg_bg, hgg_signal, 5)

    # bg dist ROOT
    bg_pdf_ROOT = functools.partial(bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 100, 180, 3)
    tf1_bg_pdf.SetParameters(*bg_result.x)

    # signal dist
    sig_pdf_ROOT = functools.partial(sig_pdf, doROOT=True)
    tf1_sig_pdf = TF1("tf1_sig_pdf", sig_pdf_ROOT, 100, 180, 2)
    tf1_sig_pdf.SetParameters(*sig_result.x)

    n_tot = n_bg + n_sig
    be_hybrid = np.concatenate([be_bg[be_bg < be_sig[0]-1.5],
                                be_sig,
                                be_bg[be_bg > be_sig[-1]+1.5]])

    be_1GeV = np.linspace(100, 180, 81)
    be_2GeV = np.linspace(100, 180, 41)
    be_5GeV = np.linspace(100, 180, 17)
    be_10GeV = np.linspace(100, 180, 9)

    true_bg_bc  = []
    true_sig_bc = []
    for i in range(len(be_hybrid)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_hybrid[i], be_hybrid[i+1])
        true_bg_bc.append(true_bg*n_bg)
        true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_result.x),
                                      be_hybrid[i], be_hybrid[i+1])
        true_sig_bc.append(true_sig*n_sig)

    template_pdf = template_pdf_wrapper(true_bg_bc, true_sig_bc)
    template_params = Parameters()
    template_params.add_many(
        ('A'    , 0.    , True  , 0    , 1    , None , None) ,
        ('n_tot' , n_tot , False , None , None , None , None)
    )


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

    for i in tnrange(1000):
        mc_bg, mc_sig = generate_toy_data(tf1_bg_pdf, tf1_sig_pdf, n_bg, n_sig)
        mc_bg_sig = mc_bg+mc_sig

        q0_nll_fit = generate_q0_via_nll_unbinned(mc_bg_sig)
        signif_nll_fit_hist.append(np.sqrt(q0_nll_fit))

        q0_nll_constrained = generate_q0_via_nll_unbinned_constrained(mc_bg, mc_bg_sig, bg_result.x)
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

        q0_bb_shape = generate_q0_via_shape_fit(mc_bg_sig, be_hybrid, template_params, template_pdf)
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
