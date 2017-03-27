#! /usr/bin/env python

from __future__ import division
import os
import functools
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from scipy.stats import poisson
import cPickle as pkl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from lmfit import Model
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from bb.tools.hist_tools_modified import hist
import nllfitter.future_fitter as ff
from ROOT import gRandom
from ROOT import TF1
from tqdm import tqdm_notebook
from bb.tools.fill_between_steps import fill_between_steps
# from bb.tools.hist_tools_modified import hist


def lm_binned_wrapper(mu_bg, mu_sig):
    def lm_binned(ix, A, ntot):
        proper_means = (((1-A)*mu_bg+A*mu_sig)/np.sum((1-A)*mu_bg+A*mu_sig))*ntot
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
    a[5]: gamma   free parameter, tqdm_notebook, tqdm_notebook, tqdm_notebook, tqdm_notebook,
    tqdm_notebook, tqdm_notebook
    '''
    if doROOT:
        x = x[0]            # this is a ROOT-compatible hack
    b = [a[3], a[4], a[5]]  # this is a ROOT-compatible hack
    c = [a[1], a[2]]        # this is a ROOT-compatible hack
    return (1 - a[0])*bg_pdf(x, b, xlow=xlow, xhigh=xhigh) + a[0]*sig_pdf(x, c)


def generate_initial_params(data_bg_mul2, data_bg_mul8):

    # fit to the data distributions
    bg_model = ff.Model(bg_pdf, ['alpha', 'beta', 'gamma'])
    bg_model.set_bounds([(-200, 200), (-100, 100), (-100, 100)])
    bg_fitter = ff.NLLFitter(bg_model, data_bg_mul2)
    bg_result = bg_fitter.fit([-1.80808e+01, -8.21174e-02, 8.06289e-01])
    n_bg = len(data_bg_mul8)

    gRandom.SetSeed(222)

    # Set up bg sampling
    bg_pdf_ROOT = functools.partial(bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 2800, 13000, 3)
    tf1_bg_pdf.SetParameters(*bg_result.x)
    mc_bg = [tf1_bg_pdf.GetRandom() for i in xrange(n_bg)]

    be_bg = bayesian_blocks(mc_bg, p0=0.02)
    be_bg = np.append(be_bg, [13000])
    be_bg[0] = 2800
    print be_bg
    # hist(data_bg_mul8, bins=be_bg, scale='binwidth')
    # plt.show()

    return bg_result, n_bg, be_bg


def generate_toy_data(bg_params, n_bg):
    '''use bg and signal params to generated simulated data'''
    # bg dist
    bg_pdf_ROOT = functools.partial(bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 2800, 13000, 3)
    tf1_bg_pdf.SetParameters(*bg_params)
    mc_bg = [tf1_bg_pdf.GetRandom() for i in xrange(n_bg)]
    return mc_bg


def calc_A_unbinned(data, model, bg_params, sig_params):
    '''Given input data and the true distribution parameters, calculate the 95% UL for the unbinned
    data.  The bg and signal parameters are held fixed.  The best-fit A value is determined first,
    then the 95% UL is determined by scanning for the correct value of A that leads to a p-value of
    0.05.  This procedure must be run many times and averaged to get the mean UL value and error
    bands.'''

    mu = sig_params[0]
    sigma = sig_params[1]
    alpha = bg_params[0]
    beta = bg_params[1]
    gamma = bg_params[2]

    # Obtain the best fit value for A
    model.set_bounds([(0, 1), (mu, mu), (sigma, sigma),
                      (alpha, alpha), (beta, beta), (gamma, gamma)])
    mle_fitter = ff.NLLFitter(model, np.asarray(data), verbose=False)
    mle_res = mle_fitter.fit([0.01, mu, sigma, alpha, beta, gamma], calculate_corr=False)

    # Now scan through A values to find the one that leads to a p-value of 0.05
    pval = -1
    right = 1
    left = mle_res.x[0]
    A_scan = 0.5*(left+right)
    while not np.isclose(pval, 0.05, 0.0001, 0.0001):
        model.set_bounds([(A_scan*(1-1e-8), A_scan*(1+1e-8)), (mu, mu), (sigma, sigma),
                          (alpha, alpha), (beta, beta), (gamma, gamma)])
        scan_fitter = ff.NLLFitter(model, np.asarray(data), verbose=False)
        scan_res = scan_fitter.fit([A_scan, mu, sigma, alpha, beta, gamma], calculate_corr=False)
        # find pval
        qu = 2*max(scan_res.fun-mle_res.fun, 0)
        pval = 1-norm.cdf(qu)
        if pval < 0.05:
            right = A_scan
            A_scan = 0.5*(right+left)
        else:
            left = A_scan
            A_scan = 0.5*(right+left)

    return scan_res.x[0]


def calc_A_binned(data, bin_edges, binned_model, params):
    '''Given input data and the true template, calculate the 95% UL for binned data
    data.  The bg and signal templates are held fixed.  The best-fit A value is determined first,
    then the 95% UL is determined by scanning for the correct value of A that leads to a p-value of
    0.05.  This procedure must be run many times and averaged to get the mean UL value and error
    bands.'''

    bc, bin_edges = np.histogram(data, bin_edges, range=(2800, 13000))
    ibc = np.asarray(range(len(bc)))
    result = binned_model.fit(bc, ix=ibc, params=params)
    # plt.plot(ibc, bc,         'bo')
    # plt.plot(ibc, result.best_fit, 'r-')
    # plt.show()
    # raw_input()
    nll_mle = -np.sum(np.log(poisson.pmf(bc, result.best_fit)))

    pval = -1
    right = 1
    left = result.params['A'].value
    A_scan = 0.5*(left+right)
    while not np.isclose(pval, 0.05, 0.0001, 0.0001):
        # find pval
        nll_scan = -np.sum(np.log(poisson.pmf(bc, result.eval(A=A_scan))))
        qu = 2*max(nll_scan-nll_mle, 0)
        pval = 1-norm.cdf(qu)
        if pval < 0.05:
            right = A_scan
            A_scan = 0.5*(right+left)
        else:
            left = A_scan
            A_scan = 0.5*(right+left)

    return A_scan


def bh_ratio_plots(data, mc, be, title='Black Hole Visual Example', save_name='bh_vis_ex',
                   do_ratio=False):
    xlims = (be[0], be[-1])
    ratlims = (0, 6)

    bin_centers = (be[1:]+be[:-1])/2

    if do_ratio:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax1.set_yscale("log", nonposy='clip')
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax1.grid(True)
        ax2.grid(True)
        # ax2.set_yscale("log", nonposy='clip')
        plt.setp(ax1.get_xticklabels(), visible=False)
        fig.subplots_adjust(hspace=0.001)
        ax1.set_xlim(xlims)
    else:
        fig, ax1 = plt.subplots()
        ax1.set_yscale("log", nonposy='clip')
        ax1.grid(True)
        ax1.set_xlim(xlims)

    # print 'lims'
    bc_d, _ = np.histogram(data, bins=be)
    bc_mc, _ = np.histogram(mc, bins=be, weights=[len(data)/(len(mc))]*(len(mc)))
    hist(data, ax=ax1, bins=be, scale='binwidth', histtype='marker', markersize=10, color='k',
         errorbars=True, label='Sim Data')

    hist(mc, ax=ax1, bins=be, scale='binwidth', weights=[len(data)/(len(mc))]*(len(mc)),
         histtype='stepfilled', alpha=0.2, label='Sim Background')
    ax1.legend()
    if do_ratio:
        ratio = bc_d/bc_mc
        ratio_err = np.sqrt(bc_d)/bc_mc
        fill_between_steps(ax2, be, ratio+ratio_err, ratio-ratio_err, alpha=0.2, step_where='pre',
                           linewidth=0, color='red')
        ax2.errorbar(bin_centers, ratio, yerr=None, xerr=[np.abs(be[0:-1]-bin_centers),
                                                          np.abs(be[1:]-bin_centers)], fmt='ok')
        ax2.set_xlabel(r'$S_T$ (GeV)', fontsize=17)
        ax2.set_ylabel('Data/BG', fontsize=17)
        ax2.get_yaxis().get_major_formatter().set_useOffset(False)
        ax2.axhline(1, linewidth=2, color='r')
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='upper'))
        ax2.set_ylim(ratlims)

    ax1.set_ylabel(r'N/$\Delta$x', fontsize=17)
    ax1.set_title(title)
    plt.savefig('figures/{}.pdf'.format(save_name))
    plt.savefig('figures/{}.png'.format(save_name))

    plt.show()


if __name__ == "__main__":
    plt.close('all')
    bb_dir  = os.path.join(os.path.dirname(__file__), '../..')
    df_data_mul2 = pkl.load(open(bb_dir+'/files/BH/BH_test_data.p', 'rb'))
    data_bg_mul2 = df_data_mul2[df_data_mul2.ST_mul2_BB >= 2800].ST_mul2_BB.values

    df_data_mul8 = pkl.load(open(bb_dir+'/files/BH/BH_paper_data.p', 'rb'))
    data_bg_mul8 = df_data_mul8[df_data_mul8.ST_mul8_BB >= 2800].ST_mul8_BB.values

    bg_result, n_bg, be_bg = generate_initial_params(data_bg_mul2, data_bg_mul8)

    true_bg_bc_bb  = []
    for i in range(len(be_bg)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_bg[i], be_bg[i+1])
        true_bg_bc_bb.append(true_bg)

    be_100GeV = np.linspace(2800, 13000, 103)
    true_bg_bc_100GeV  = []
    true_sig_bc_100GeV = []
    for i in range(len(be_100GeV)-1):
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_result.x),
                                      be_100GeV[i], be_100GeV[i+1])
        true_bg_bc_100GeV.append(true_bg)
    #     true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=(5000, 1000)),
    #                                   be_100GeV[i], be_100GeV[i+1])
    #     true_sig_bc_100GeV.append(true_sig)

    # Do a bunch of toys
    gRandom.SetSeed(20)

    unbinned_A = [[], [], [], []]
    binned_A = [[], [], [], []]
    binned_A_100 = [[], [], [], []]
    bg_sig_model = ff.Model(bg_sig_pdf, ['C', 'mu', 'sigma', 'alpha', 'beta', 'gamma'])
    sig_params = [(4000, 800), (5000, 1000), (6000, 1200), (7000, 1400)]

    # mc_bg = generate_toy_data(bg_result.x, n_bg)
    # res = calc_A_binned(mc_bg, be_bg, binned_model, binned_params)
    for i, sig_p in enumerate(tqdm_notebook(sig_params, desc='Signal Model')):

        # Set up binned model for BB
        true_sig_bc_bb = []
        for k in range(len(be_bg)-1):
            true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_p),
                                          be_bg[k], be_bg[k+1])
            true_sig_bc_bb.append(true_sig)

        lm_binned_bb = lm_binned_wrapper(np.asarray(true_bg_bc_bb), np.asarray(true_sig_bc_bb))
        binned_model_bb = Model(lm_binned_bb)
        binned_params_bb = binned_model_bb.make_params()
        binned_params_bb['ntot'].value   = n_bg
        binned_params_bb['ntot'].vary    = False
        binned_params_bb['A'].value      = 0.1
        binned_params_bb['A'].min        = 0
        binned_params_bb['A'].max        = 1

        # Set up binned model for 100 GeV
        true_sig_bc_100GeV = []
        for k in range(len(be_100GeV)-1):
            true_sig, _  = integrate.quad(functools.partial(sig_pdf, a=sig_p),
                                          be_100GeV[k], be_100GeV[k+1])
            true_sig_bc_100GeV.append(true_sig)

        lm_binned_100GeV = lm_binned_wrapper(np.asarray(true_bg_bc_100GeV),
                                             np.asarray(true_sig_bc_100GeV))
        binned_model_100GeV = Model(lm_binned_100GeV)
        binned_params_100GeV = binned_model_100GeV.make_params()
        binned_params_100GeV['ntot'].value   = n_bg
        binned_params_100GeV['ntot'].vary    = False
        binned_params_100GeV['A'].value      = 0.1
        binned_params_100GeV['A'].min        = 0
        binned_params_100GeV['A'].max        = 1

        for j in tqdm_notebook(xrange(200), desc='Toys', leave=False):
            mc_bg = generate_toy_data(bg_result.x, n_bg)
            uA = calc_A_unbinned(mc_bg, bg_sig_model, bg_result.x, sig_p)
            unbinned_A[i].append(uA)
            bA = calc_A_binned(mc_bg, be_bg, binned_model_bb, binned_params_bb)
            binned_A[i].append(bA)
            bA_100 = calc_A_binned(mc_bg, be_100GeV, binned_model_100GeV, binned_params_100GeV)
            binned_A_100[i].append(bA_100)
