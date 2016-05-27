#! /usr/bin/env python

from __future__ import division
import os
import functools
import bisect
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import cPickle as pkl
from matplotlib import pyplot as plt
# from bb.tools.bayesian_blocks_modified import bayesian_blocks
from bb.tools.bb_plotter import make_fit_plot, make_bb_plot
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
    print ''
    print 'Calculating local p-value and significance based on {0} toys'.format(ntoys)
    print 'N_bg = {0}, sigma_bg = {1}, N_signal = {2}'.format(N_bg,var_bg,N_sig)
    toys    = np.random.normal(N_bg, var_bg, int(ntoys))
    #print toys
    pvars   = np.random.poisson(toys)
    pval    = pvars[pvars > N_bg + N_sig].size/ntoys
    print 'local p-value = {0}'.format(pval)
    print 'local significance = {0:.2f}'.format(np.abs(norm.ppf(1-pval)))


if __name__ == "__main__":
    plt.close('all')
    current_dir = os.path.dirname(__file__)
    bb_dir      = os.path.join(current_dir, '../..')
    hgg_bg      = pkl.load(open(bb_dir+'/files/hgg_bg.p', "rb"))
    hgg_signal  = pkl.load(open(bb_dir+'/files/hgg_signal.p', "rb"))

# grab a handful of bg events, and an ~X sigma number of signal events
    n_sigma              = 5
    hgg_bg_selection     = hgg_bg[(hgg_bg.Mgg > 100) & (hgg_bg.Mgg < 180)][0:10000].Mgg
    n_bg_under_sig       = hgg_bg_selection[(118 < hgg_bg_selection) & (hgg_bg_selection < 133)].size
    n_sig                = int(n_sigma*np.sqrt(n_bg_under_sig))
    hgg_signal_selection = hgg_signal[(hgg_signal.Mgg >= 118) & (hgg_signal.Mgg <= 133)][0:n_sig].Mgg
    data_bg              = hgg_bg_selection.values
    data_sig             = hgg_signal_selection.values
    data_bg_sig          = np.concatenate((data_bg, data_sig))

    print 'loaded'

# setup initial ranges and binning
    xlimits = (100., 180.)
    x       = np.linspace(xlimits[0], xlimits[1], 10000)
    nbins   = 80
    binning = (xlimits[1]-xlimits[0])/nbins

# fit to the data distributions
    bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
    bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])
    bg_fitter = ff.NLLFitter(bg_model, data_bg)
    bg_result = bg_fitter.fit([0.0, 0.0, 0.0])

    sig_model = ff.Model(sig_pdf, ['mu', 'sigma'])
    sig_model.set_bounds([(110, 130), (1, 5)])
    sig_fitter = ff.NLLFitter(sig_model, data_sig)
    sig_result = sig_fitter.fit([120.0, 2])

    bg_sig_model = ff.Model(bg_sig_pdf, ['C', 'mu', 'sigma', 'a1', 'a2', 'a3'])
    bg_sig_model.set_bounds([(0, 1), (xlimits[0], xlimits[1]), (0, 50), (-1., 1.), (-1., 1.), (-1., 1.)])
    bg_sig_fitter = ff.NLLFitter(bg_sig_model, data_bg_sig)
    bg_sig_result = bg_sig_fitter.fit([0.01, 125.0, 2, 0.0, 0.0, 0.0])

# make MC
    gRandom.SetSeed(111)
# bg dist
    bg_pdf_ROOT = functools.partial(bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 100, 180, 3)
    tf1_bg_pdf.SetParameters(*bg_result.x)
# signal dist
    sig_pdf_ROOT = functools.partial(sig_pdf, doROOT=True)
    tf1_sig_pdf = TF1("tf1_sig_pdf", sig_pdf_ROOT, 100, 180, 2)
    tf1_sig_pdf.SetParameters(*sig_result.x)
    mc_bg = [tf1_bg_pdf.GetRandom() for i in xrange(len(data_bg))]
    mc_sig = [tf1_sig_pdf.GetRandom() for i in xrange(len(data_sig))]
    mc_bg_sig = mc_bg+mc_sig
    mc_bg_sig_fitter = ff.NLLFitter(bg_sig_model, np.asarray(mc_bg_sig))
    mc_bg_sig_result = mc_bg_sig_fitter.fit([0.01, 125.0, 2, 0.0, 0.0, 0.0])
    mc_bg_bad_fitter = ff.NLLFitter(bg_model, np.asarray(mc_bg_sig))
    mc_bg_bad_result = mc_bg_bad_fitter.fit([ 0.0, 0.0, 0.0])

    n_bg_calc, _  = integrate.quad(functools.partial(bg_sig_pdf, a=np.concatenate([[0], mc_bg_sig_result.x[1:]])),
            mc_bg_sig_result.x[1]-mc_bg_sig_result.x[2]*2, mc_bg_sig_result.x[1]+mc_bg_sig_result.x[2]*2)
    n_bg_calc *= len(mc_bg_sig)*(1-mc_bg_sig_result.x[0])
    mc_bg_sig_result.x[1]-mc_bg_sig_result.x[2]*2, mc_bg_sig_result.x[1]+mc_bg_sig_result.x[2]*2
    n_sig_calc = len(mc_bg_sig)*mc_bg_sig_result.x[0]
    n_sig_std = mc_bg_sig_fitter.get_corr(mc_bg_sig_result.x)[0][0]
    print n_sig_calc, n_sig_std*len(mc_bg_sig), n_bg_calc, np.sqrt(n_bg_calc)
    calc_local_pvalue(n_bg_calc, np.sqrt(n_bg_calc), n_sig, n_sig_std*len(mc_bg_sig),1e7)


# plot some outputs

#make_fit_plot(data_bg, 80, xlimits, functools.partial(bg_pdf, a=bg_result.x),
#        'Background distribution')
#make_fit_plot(data_sig, 80, xlimits, functools.partial(sig_pdf, a=sig_result.x),
#        'Signal distribution')
#make_fit_plot(data_bg_sig, 80, xlimits, functools.partial(bg_sig_pdf, a=bg_sig_result.x),
#        'Signal+Background distribution', extra_pdf_tuple=(functools.partial(bg_sig_pdf,
#            a=np.concatenate([[0], bg_sig_result.x[1:]])), 1-bg_sig_result.x[0], 'bg pdf'))
    textstr = ('$C={0:.2f}$\n$\mu={1:.2f}$\n$\sigma={2:.2f}$\n'
                '$a_1={3:.2f}$\n$a_2={4:.2f}$\n$a_3={5:.2f}$').format(*mc_bg_sig_result.x)
    make_fit_plot(mc_bg_sig, 80, xlimits, functools.partial(bg_sig_pdf, a=mc_bg_sig_result.x),
            'Signal+Background MC Toy', extra_pdf_tuple=(functools.partial(bg_sig_pdf,
                a=np.concatenate([[0], mc_bg_sig_result.x[1:]])), 1-mc_bg_sig_result.x[0], 'bg pdf'), textstr = textstr)
#make_bb_plot(data_bg_sig, 0.02, bb_dir+'/plots/', range=xlimits, title=r'pp$\to\gamma\gamma$ Sim',
#             xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin', save_name='hgg_inject_hist')
#make_bb_plot(data_bg, 0.02, bb_dir+'/plots/', range=xlimits, title=r'pp$\to\gamma\gamma$ Sim',
#             xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin', save_name='hgg_bg_hist')
#make_bb_plot(data_sig, 0.02, bb_dir+'/plots/', range=xlimits, title=r'pp$\to\gamma\gamma$ Sim',
#        xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin', save_name='hgg_sig_hist')
    bc_bg, be_bg = make_bb_plot(mc_bg, 0.02, bb_dir+'/plots/', range=xlimits, title='BB binning for Background Distribution',
                   xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin', save_name='hgg_bg_hist')
    bc_sig, be_sig = make_bb_plot(mc_sig, 0.02, bb_dir+'/plots/', range=xlimits, title='BB binning for Signal Distribution',
                     xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin', save_name='hgg_sig_hist')


    plt.show()

# convert be_sig to widths:


# run bb on many MC toys

#bg_bc = {}
#bg_be = {}
#bg_sig_bc = {}
#bg_sig_be = {}
#for toy in range(100):
#    print 'toy', toy
#    mc_bg                                = [tf1_bg_pdf.GetRandom() for i in xrange(len(data_bg))]
#    mc_sig                               = [tf1_sig_pdf.GetRandom() for i in xrange(len(data_sig))]
#    mc_bg_sig                            = mc_bg+mc_sig
#    bg_bin_content, bg_bin_edges         = np.histogram(mc_bg, bayesian_blocks(mc_bg, p0=0.02), density=True)
#    bg_sig_bin_content, bg_sig_bin_edges = np.histogram(mc_bg_sig, bayesian_blocks(mc_bg_sig, p0=0.02), density=True)
#    bg_bc['toy'+str(toy)]                = bg_bin_content
#    bg_be['toy'+str(toy)]                = bg_bin_edges
#    bg_sig_bc['toy'+str(toy)]            = bg_sig_bin_content
#    bg_sig_be['toy'+str(toy)]            = bg_sig_bin_edges

# do stuff with these bcs and bes
