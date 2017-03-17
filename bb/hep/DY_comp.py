#! /usr/bin/env python

from __future__ import division
import os
import cPickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from bb.tools.hist_tools_modified import hist
from bb.tools.fill_between_steps import fill_between_steps


def fancy_plots(bin_style='bb'):
    current_dir = os.path.dirname(__file__)
    bb_dir = os.path.join(current_dir, '../..')
    xlims = (50, 140)
    ratlims = (0, 3)
    n_events = 10000

    z_data = pkl.load(open(bb_dir+'/files/DY/ZLL_v2.p', "rb"))
    z_data = z_data.query('50<Mll<140')

    if bin_style == 'bb':
        be = bayesian_blocks(z_data[0:n_events].Mll, p0=0.01)
        be[0] = xlims[0]
        be[-1] = xlims[1]
        bin_centers = (be[1:]+be[:-1])/2
        print len(bin_centers)
    else:
        _, be = np.histogram(z_data[0:n_events].Mll, 20, range=xlims)
        bin_centers = (be[1:]+be[:-1])/2

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax1.set_yscale("log", nonposy='clip')
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0.001)
    ax1.set_xlim(xlims)

    print 'lims'
    print z_data.Mll.min(), z_data.Mll.max()
    bc_d, _, _  = hist(z_data[0:n_events].Mll*1.01, ax=ax1, bins=be, scale='binwidth',
                       histtype='marker', markersize=10, color='k', errorbars=True, label='Shifted')

    bc_mc, _, _ = hist(z_data[n_events:].Mll, ax=ax1, bins=be, scale='binwidth',
                       weights=[n_events/(len(z_data)-n_events)]*(len(z_data)-n_events),
                       histtype='stepfilled', alpha=0.2, label='Nominal')
    # bc_mc, _, _ = hist(z_data.Mll_gen, ax=ax1, bins=be, scale='binwidth',
    #                    weights=[n_events/len(z_data)]*len(z_data),
    #                    histtype='stepfilled', alpha=0.2, label='Gen')
    ax1.legend()
    ratio = bc_d/bc_mc
    ratio_err = np.sqrt(bc_d)/bc_mc
    fill_between_steps(ax2, be, ratio+ratio_err, ratio-ratio_err, alpha=0.2, step_where='pre',
                       linewidth=0, color='red')
    ax2.errorbar(bin_centers, ratio, yerr=None, xerr=[np.abs(be[0:-1]-bin_centers),
                                                      np.abs(be[1:]-bin_centers)], fmt='ok')
    ax2.set_xlabel(r'$M_{\mu\mu}$ (GeV)', fontsize=17)
    ax2.set_ylabel('Data/BG', fontsize=17)
    ax1.set_ylabel(r'N/$\Delta$x', fontsize=17)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.axhline(1, linewidth=2, color='r')
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='upper'))
    ax2.set_ylim(ratlims)
    if bin_style == 'bb':
        ax1.set_title('Dimuon Mass Distribution, Bayesian Blocks')
        plt.savefig('figures/bb_Z_shifted.pdf')
        plt.savefig('figures/bb_Z_shifted.png')
    else:
        ax1.set_title('Dimuon Mass Distribution, Uniform Bins')
        plt.savefig('figures/b25_Z_shifted.pdf')
        plt.savefig('figures/b25_Z_shifted.png')

    plt.show()

def fancy_plots_pt(bin_style='bb'):
    current_dir = os.path.dirname(__file__)
    bb_dir = os.path.join(current_dir, '../..')
    pt_data = pkl.load(open(bb_dir+'/files/DY/ZLL_Jet1.p', "rb"))
    pt_data = pt_data.query('0<leading_jet_pT').reset_index().ix[0:9000]
    pt_mc = pkl.load(open(bb_dir+'/files/DY/ZLL_Jet2.p', "rb"))
    pt_mc = pt_mc.query('0<leading_jet_pT')
    xlims = (pt_data.leading_jet_pT.min(), pt_data.leading_jet_pT.max())
    ratlims = (0, 3)
    n_events = len(pt_data)


    if bin_style == 'bb':
        be = bayesian_blocks(pt_data.leading_jet_pT, p0=0.01)
        be[0] = xlims[0]
        be[-1] = xlims[1]
        bin_centers = (be[1:]+be[:-1])/2
        print len(bin_centers)
    else:
        _, be = np.histogram(pt_data.leading_jet_pT, 15, range=xlims)
        bin_centers = (be[1:]+be[:-1])/2

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax1.set_yscale("log", nonposy='clip')
    ax1.set_xscale("log", nonposy='clip')
    ax1.set_xticks([30, 50, 100, 200, 400, 800])
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0.001)
    ax1.set_xlim(xlims)

    print 'lims'
    print pt_data.leading_jet_pT.min(), pt_data.leading_jet_pT.max()
    bc_d, _, _  = hist(pt_data.leading_jet_pT*1.0, ax=ax1, bins=be, scale='binwidth',
                       histtype='marker', markersize=10, color='k', errorbars=True, label='Data')

    bc_mc, _, _ = hist(pt_mc.leading_jet_pT, ax=ax1, bins=be, scale='binwidth',
                       weights=[n_events/len(pt_mc)]*(len(pt_mc)),
                       histtype='stepfilled', alpha=0.2, label='MC')
    # bc_mc, _, _ = hist(z_data.Mll_gen, ax=ax1, bins=be, scale='binwidth',
    #                    weights=[n_events/len(z_data)]*len(z_data),
    #                    histtype='stepfilled', alpha=0.2, label='Gen')
    ax1.legend()
    ratio = bc_d/bc_mc
    ratio_err = np.sqrt(bc_d)/bc_mc
    fill_between_steps(ax2, be, ratio+ratio_err, ratio-ratio_err, alpha=0.2, step_where='pre',
                       linewidth=0, color='red')
    ax2.errorbar(bin_centers, ratio, yerr=None, xerr=[np.abs(be[0:-1]-bin_centers),
                                                      np.abs(be[1:]-bin_centers)], fmt='ok')
    ax2.set_xlabel(r'Jet $p_T$ (GeV)', fontsize=17)
    ax2.set_ylabel('Data/BG', fontsize=17)
    ax1.set_ylabel(r'N/$\Delta$x', fontsize=17)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.axhline(1, linewidth=2, color='r')
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='upper'))
    ax2.set_ylim(ratlims)
    if bin_style == 'bb':
        ax1.set_title('Leading Jet Distribution, Bayesian Blocks')
        plt.savefig('figures/bb_Jet_comp.pdf')
        plt.savefig('figures/bb_Jet_comp.png')
    else:
        ax1.set_title('Leading Jet Distribution, Uniform Bins')
        plt.savefig('figures/b25_Jet_comp.pdf')
        plt.savefig('figures/b25_Jet_comp.png')

    plt.show()


def simple_binning():
    plt.close('all')
    current_dir = os.path.dirname(__file__)
    bb_dir = os.path.join(current_dir, '../..')
    xlims = (30, 200)

    z_data = pkl.load(open(bb_dir+'/files/DY/Linear.p', "rb"))
    n_events = 100000
    be = bayesian_blocks(z_data[0:n_events].Mll_gen, p0=0.01)
    print be
    hist(z_data[0:n_events].Mll_gen, bins=170, scale='binwidth', histtype='stepfilled', alpha=0.2,
         label='1 GeV', range=xlims)
    hist(z_data[0:n_events].Mll_gen, bins=be, scale='binwidth', histtype='step', color='red',
         label='BB', range=xlims)
    plt.show()

if __name__ == "__main__":
    # simple_binning()
    plt.close('all')
    #fancy_plots(25)
    #fancy_plots('bb')
    fancy_plots_pt(25)
    fancy_plots_pt('bb')
