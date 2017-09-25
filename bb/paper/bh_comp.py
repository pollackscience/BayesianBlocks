#! /usr/bin/env python

from __future__ import division
from __future__ import absolute_import
import os
import functools
import numpy as np
# from scipy.stats import norm
import scipy.integrate as integrate
from scipy.stats import poisson
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from histogram_plus import hist
from nllfit import NLLFitter, Model
from lmfit import Parameters
from ROOT import gRandom
from ROOT import TF1
from tqdm import tqdm_notebook
from bb.tools.fill_between_steps import fill_between_steps
from numba import jit
from six.moves import range
import pandas as pd
# from bb.tools.hist_tools_modified import hist
bb_dir  = '/Users/brianpollack/Coding/BayesianBlocks'


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
    # if not cnc:
    #     mu_bg = mu_bg/float(np.sum(mu_bg))
    #     mu_sig = mu_sig/float(np.sum(mu_sig))

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
@jit
def bg_pdf_helper(x, a0, a1, a2):
    return ((1+x)**a0)/(x**(a1+a2*np.log(x)))
    # return ((1+x)**a0)/(x**(a2*np.log(x)))


@jit
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
    # print x, a
    return func(x)/max(integrate.quad(func, 2800, 13000)[0], 1e-200)


@jit
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
    a[5]: gamma   free parameter,
    '''
    if doROOT:
        x = x[0]            # this is a ROOT-compatible hack
    b = [a[3], a[4], a[5]]  # this is a ROOT-compatible hack
    c = [a[1], a[2]]        # this is a ROOT-compatible hack
    return (1 - a[0])*bg_pdf(x, b, xlow=xlow, xhigh=xhigh) + a[0]*sig_pdf(x, c)


def generate_initial_params(data_bg_mul2, data_bg_mul8, seed=5):

    # fit to the data distributions

    bg_params = Parameters()
    bg_params.add_many(
        ('alpha', -1.80808e+01, True, 1e-20, 20, None, None),
        ('beta', -8.21174e-02, True, -10, -1e-20, None, None),
        ('gamma', 8.06289e-01, True, 1e-20, 10, None, None)
    )

    bg_model = Model(bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model)
    bg_result = bg_fitter.fit(data_bg_mul2, calculate_corr=False)

    n_bg = len(data_bg_mul8)

    gRandom.SetSeed(seed)

    # Set up bg sampling
    bg_pdf_ROOT = functools.partial(bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 2800, 13000, 3)
    tf1_bg_pdf.SetParameters(*bg_result.x)
    mc_bg = [tf1_bg_pdf.GetRandom() for i in range(n_bg)]

    be_bg = bayesian_blocks(mc_bg, p0=0.02)
    be_bg[-1] += 0.1
    be_bg = np.append(be_bg, [13000])
    be_bg[0] = 2800
    # print be_bg
    # hist(data_bg_mul8, bins=be_bg, scale='binwidth')
    # plt.show()

    return bg_result, n_bg, be_bg


def generate_toy_data_wrapper(bg_params):
    '''use bg and signal params to generated simulated data'''
    # bg dist
    bg_pdf_ROOT = functools.partial(bg_pdf, doROOT=True)
    tf1_bg_pdf = TF1("tf1_bg_pdf", bg_pdf_ROOT, 2800, 13000, 3)
    tf1_bg_pdf.SetParameters(*bg_params)

    def generate_toy_data(n_bg):
        mc_bg = [tf1_bg_pdf.GetRandom() for i in range(n_bg)]
        return mc_bg
    return generate_toy_data


def calc_A_unbinned(data, bg_params, sig_params):
    '''Given input data and the true distribution parameters, calculate the 95% UL for the unbinned
    data.  The bg and signal parameters are held fixed.  The best-fit A value is determined first,
    then the 95% UL is determined by scanning for the correct value of A that leads to a p-value of
    0.05.  This procedure must be run many times and averaged to get the mean UL value and error
    bands.'''

    mu    = sig_params[0]
    sigma = sig_params[1]
    alpha = bg_params[0]
    beta  = bg_params[1]
    gamma = bg_params[2]

    params = Parameters()
    params.add_many(
        ('C'     , 0.01  , True  , 0    , 1    , None , None) ,
        ('mu'    , mu    , False , None , None , None , None) ,
        ('sigma' , sigma , False , None , None , None , None) ,
        ('alpha' , alpha , False , None , None , None , None) ,
        ('beta'  , beta  , False , None , None , None , None) ,
        ('gamma' , gamma , False , None , None , None , None)
    )

    bg_sig_model = Model(bg_sig_pdf, params)

    # Obtain the best fit value for A
    mle_fitter = NLLFitter(bg_sig_model)
    mle_res = mle_fitter.fit(np.asarray(data), calculate_corr=False,
                             verbose=False)

    return mle_res.x[0]


def calc_A_binned(data, bg_mu, sig_mu):
    '''Given input data and the true template, calculate the 95% UL for binned data
    data.  The bg and signal templates are held fixed.  The best-fit A value is determined first,
    then the 95% UL is determined by scanning for the correct value of A that leads to a p-value of
    0.05.  This procedure must be run many times and averaged to get the mean UL value and error
    bands.'''

    # Set up the models and pdfs, given the true means
    n_tot = np.sum(data)

    template_pdf = template_pdf_wrapper(bg_mu, sig_mu)
    template_params = Parameters()
    template_params.add_many(
        ('A'    , 0.1    , True  , 0    , 1    , None , None) ,
        ('n_tot' , n_tot , False , None , None , None , None)
    )

    template_model = Model(template_pdf, template_params)

    # Obtain the best fit value for A
    template_fitter = NLLFitter(template_model)
    mle_res = template_fitter.fit(data, calculate_corr=False, verbose=False)

    return mle_res.x[0]


def calc_A_cnc(data, bg_params, sig_params, xlow=2800, cache_true=None, cache_fit=None):
    '''Given input data and the true template, calculate the 95% UL for a single binned
    data.  The bg and signal templates are held fixed.  The best-fit A value is determined first,
    then the 95% UL is determined by scanning for the correct value of A that leads to a p-value of
    0.05.  This procedure must be run many times and averaged to get the mean UL value and error
    bands.'''
    if cache_true is None:
        cache_true = {}
    if cache_fit is None:
        cache_fit = {}

    # Set up the models and pdfs, given the true means
    data = np.asarray(data)

    if xlow in cache_true:
        true_bg, true_sig = cache_true[xlow]
    else:
        true_bg, _   = integrate.quad(functools.partial(bg_pdf, a=bg_params), xlow, 13000)
        true_sig, _   = integrate.quad(functools.partial(sig_pdf, a=sig_params), xlow, 13000)
        cache_true[xlow] = (true_bg, true_sig)

    tmp_data = data[data > xlow]
    # if len(tmp_data) is 0:
    #     raise Exception('no data after cut={}'.format(xlow))
    if len(tmp_data) in cache_fit and xlow in cache_true:
        mle_a = cache_fit[len(tmp_data)]
    else:

        n_tot = len(data)
        template_pdf = template_pdf_wrapper([true_bg], [true_sig], cnc=True)

        template_params = Parameters()
        template_params.add_many(
            ('A'    , 0.1    , True  , 0    , 1    , None , None) ,
            ('n_tot' , n_tot , False , None , None , None , None)
        )

        template_model = Model(template_pdf, template_params)
        template_fitter = NLLFitter(template_model)

        # Obtain the best fit value for A
        ntmp = len(tmp_data)
        if ntmp < 3:
            ntmp = 3
        mle_res = template_fitter.fit(np.asarray([ntmp]), calculate_corr=False, verbose=False)
        mle_a = mle_res.x[0]
        cache_fit[len(tmp_data)] = mle_a

    return mle_a, cache_true, cache_fit


def bh_ratio_plots(data, mc, be, title='Black Hole Visual Example', save_name='bh_vis_ex',
                   do_ratio=False, signal_mc=None, n_sig=10):
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
    hist(data, ax=ax1, bins=be, scale='binwidth', histtype='marker', color='k',
         errorbars=True, suppress_zero=True, label='Sim Data', err_x=False, err_type='poisson')

    hist(mc, ax=ax1, bins=be, scale='binwidth', weights=[len(data)/(len(mc))]*(len(mc)),
         histtype='stepfilled', alpha=0.4, label='Sim Background')
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
    if not isinstance(signal_mc, type(None)):
        hist(signal_mc, ax=ax1, bins=be, scale='binwidth',
             weights=[n_sig/(len(signal_mc))]*(len(signal_mc)),
             histtype='step', linestyle='--', linewidth=3, label='Sim Signal', color='xkcd:pumpkin')

    ax1.legend()
    ax1.set_ylabel(r'$\mathrm{\frac{Counts}{Bin\ Width}}$', fontsize=30)
    ax1.set_xlabel('$S_T$ (GeV)', fontsize=25)
    ax1.set_title(title, fontsize=25)

    plt.savefig(bb_dir+'/plots/bh_example/{}.pdf'.format(save_name))

    plt.show()


def get_true_bin_content(be, pdf, params):
    true_bc_list = []
    for k in range(len(be)-1):
        true_bc, _   = integrate.quad(functools.partial(pdf, a=params),
                                      be[k], be[k+1])
        true_bc_list.append(true_bc)
    return true_bc_list


if __name__ == "__main__":
    seed = 26
    plt.close('all')
    np.random.seed(seed)
    bb_dir  = os.path.join(os.path.dirname(__file__), '../..')
    df_data_mul2 = pd.read_pickle(bb_dir+'/files/BH/BH_test_data.p')
    data_bg_mul2 = df_data_mul2[df_data_mul2.ST_mul2_BB >= 2800].ST_mul2_BB.values

    df_data_mul8 = pd.read_pickle(bb_dir+'/files/BH/BH_paper_data.p')
    data_bg_mul8 = df_data_mul8[df_data_mul8.ST_mul8_BB >= 2800].ST_mul8_BB.values

    bg_result, n_bg, be_bg = generate_initial_params(data_bg_mul2, data_bg_mul8, seed)
    gen_toy = generate_toy_data_wrapper(bg_result.x)

    be_50GeV = np.linspace(2800, 13000, 205)
    be_100GeV = np.linspace(2800, 13000, 103)
    be_200GeV = np.linspace(2800, 13000, 52)
    be_400GeV = np.linspace(2800, 13000, 26)
    be_1000GeV = np.linspace(2800, 13000, 11)
    be_2000GeV = np.linspace(2800, 13000, 6)

    true_bg_bc_bb      = get_true_bin_content(be_bg, bg_pdf, bg_result.x)
    true_bg_bc_50GeV   = get_true_bin_content(be_50GeV, bg_pdf, bg_result.x)
    true_bg_bc_100GeV  = get_true_bin_content(be_100GeV, bg_pdf, bg_result.x)
    true_bg_bc_200GeV  = get_true_bin_content(be_200GeV, bg_pdf, bg_result.x)
    true_bg_bc_400GeV  = get_true_bin_content(be_400GeV, bg_pdf, bg_result.x)
    true_bg_bc_1000GeV = get_true_bin_content(be_1000GeV, bg_pdf, bg_result.x)
    true_bg_bc_2000GeV = get_true_bin_content(be_2000GeV, bg_pdf, bg_result.x)

    # Do a bunch of toys
    gRandom.SetSeed(seed)

    # bg_sig_model = ff.Model(bg_sig_pdf, ['C', 'mu', 'sigma', 'alpha', 'beta', 'gamma'])
    # sig_params = [(4000, 800), (5000, 1000), (6000, 1200), (7000, 1400)]
    # sig_params = [(4750, 970), (5350, 1070), (6000, 1200), (6600, 1300),
    #               (7150, 1440), (7800, 1500), (8380, 1660)]
    sig_params = [(5350, 1070), (6000, 1200), (6600, 1300),
                  (7150, 1440), (7800, 1500), (8380, 1660)]
    # sig_params = [(7150, 1440)]

    unbinned_A_mle      = [[] for i in range(len(sig_params))]
    binned_A_mle        = [[] for i in range(len(sig_params))]
    binned_A_hybrid_mle = [[] for i in range(len(sig_params))]
    binned_A_50_mle     = [[] for i in range(len(sig_params))]
    binned_A_100_mle    = [[] for i in range(len(sig_params))]
    binned_A_200_mle    = [[] for i in range(len(sig_params))]
    binned_A_400_mle    = [[] for i in range(len(sig_params))]
    binned_A_1000_mle   = [[] for i in range(len(sig_params))]
    binned_A_2000_mle   = [[] for i in range(len(sig_params))]
    cnc_A_mle           = [[] for i in range(len(sig_params))]

    sig_pdf_ROOT = functools.partial(sig_pdf, doROOT=True)
    tf1_sig_pdf = TF1("tf1_sig_pdf", sig_pdf_ROOT, 2800, 13000, 2)

    for i, sig_p in enumerate(tqdm_notebook(sig_params, desc='Signal Model')):

        n_sig = n_bg
        tf1_sig_pdf.SetParameters(*sig_p)
        mc_sig = [tf1_sig_pdf.GetRandom() for ns in range(n_sig)]
        be_sig = bayesian_blocks(mc_sig, p0=0.02)

        true_sig_bc_bb      = get_true_bin_content(be_bg, sig_pdf, sig_p)
        true_sig_bc_50GeV   = get_true_bin_content(be_50GeV, sig_pdf, sig_p)
        true_sig_bc_100GeV  = get_true_bin_content(be_100GeV, sig_pdf, sig_p)
        true_sig_bc_200GeV  = get_true_bin_content(be_200GeV, sig_pdf, sig_p)
        true_sig_bc_400GeV  = get_true_bin_content(be_400GeV, sig_pdf, sig_p)
        true_sig_bc_1000GeV = get_true_bin_content(be_1000GeV, sig_pdf, sig_p)
        true_sig_bc_2000GeV = get_true_bin_content(be_2000GeV, sig_pdf, sig_p)

        be_hybrid = np.sort(np.unique(np.concatenate([be_bg, be_sig])))

        true_bg_bc_bb_hybrid  = get_true_bin_content(be_hybrid, bg_pdf, bg_result.x)
        true_sig_bc_bb_hybrid = get_true_bin_content(be_hybrid, sig_pdf, sig_p)

        # for j in xrange(1000):
        for j in tqdm_notebook(range(3000), desc='Toys', leave=False):
            mc_bg = gen_toy(np.random.poisson(n_bg))
            # mc_bg = gen_toy(n_bg)
            # Per toy bb binning:

            # be_bg = bayesian_blocks(mc_bg, p0=0.02)
            # be_bg[-1] += 0.1
            # be_bg = np.append(be_bg, [13000])
            # be_bg[0] = 2800
            # be_hybrid = np.sort(np.unique(np.concatenate([be_bg, be_sig])))

            # true_bg_bc_bb  = get_true_bin_content(be_bg, bg_pdf, bg_result.x)
            # true_sig_bc_bb = get_true_bin_content(be_bg, sig_pdf, sig_p)
            # true_bg_bc_bb_hybrid  = get_true_bin_content(be_hybrid, bg_pdf, bg_result.x)
            # true_sig_bc_bb_hybrid = get_true_bin_content(be_hybrid, sig_pdf, sig_p)

            uA_mle = calc_A_unbinned(mc_bg, bg_result.x, sig_p)
            unbinned_A_mle[i].append(uA_mle)

            cA_mle, _, _ = calc_A_cnc(mc_bg, bg_result.x, sig_p, xlow=4800)
            cnc_A_mle[i].append(cA_mle)

            bc_bb, _ = np.histogram(mc_bg, bins=be_bg)
            bA_mle = calc_A_binned(bc_bb, true_bg_bc_bb, true_sig_bc_bb)
            binned_A_mle[i].append(bA_mle)

            bc_hybrid, _ = np.histogram(mc_bg, bins=be_hybrid)
            bA_hybrid_mle = calc_A_binned(bc_hybrid, true_bg_bc_bb_hybrid, true_sig_bc_bb_hybrid)
            binned_A_hybrid_mle[i].append(bA_hybrid_mle)

            bc_50, _ = np.histogram(mc_bg, bins=be_50GeV)
            bA_50_mle = calc_A_binned(bc_50, true_bg_bc_50GeV, true_sig_bc_50GeV)
            binned_A_50_mle[i].append(bA_50_mle)

            bc_100, _ = np.histogram(mc_bg, bins=be_100GeV)
            bA_100_mle = calc_A_binned(bc_100, true_bg_bc_100GeV, true_sig_bc_100GeV)
            binned_A_100_mle[i].append(bA_100_mle)

            bc_200, _ = np.histogram(mc_bg, bins=be_200GeV)
            bA_200_mle = calc_A_binned(bc_200, true_bg_bc_200GeV, true_sig_bc_200GeV)
            binned_A_200_mle[i].append(bA_200_mle)

            bc_400, _ = np.histogram(mc_bg, bins=be_400GeV)
            bA_400_mle = calc_A_binned(bc_400, true_bg_bc_400GeV, true_sig_bc_400GeV)
            binned_A_400_mle[i].append(bA_400_mle)

            bc_1000, _ = np.histogram(mc_bg, bins=be_1000GeV)
            bA_1000_mle = calc_A_binned(bc_1000, true_bg_bc_1000GeV, true_sig_bc_1000GeV)
            binned_A_1000_mle[i].append(bA_1000_mle)

            bc_2000, _ = np.histogram(mc_bg, bins=be_2000GeV)
            bA_2000_mle = calc_A_binned(bc_2000, true_bg_bc_2000GeV, true_sig_bc_2000GeV)
            binned_A_2000_mle[i].append(bA_2000_mle)
