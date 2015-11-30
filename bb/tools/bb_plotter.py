#! /usr/bin/env python

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from hist_tools_modified import hist
from bayesian_blocks_modified import bayesian_blocks
from fill_between_steps import fill_between_steps
from matplotlib.ticker import MaxNLocator


def make_hist_ratio_blackhole(bin_edges, data, mc, data_err, label, suffix = None, data_driven=True, signal=None, mode='no_signal'):
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
    ax1=fig.add_subplot(gs[0])
    ax2=fig.add_subplot(gs[1],sharex=ax1)
    ax1.grid(True)
    ax2.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.subplots_adjust(hspace=0.001)
    #ax = plt.gca()
    ax1.set_yscale("log", nonposy='clip')
    fill_between_steps(ax1, bin_edges, mc,1e-4, alpha=0.2, step_where='pre',linewidth=0,label='QCD MC')
    if mode in ['signal_search','signal_search_inj']:
        fill_between_steps(ax1, bin_edges,mc+signal,mc,alpha=0.6,step_where='pre',linewidth=0,label='Signal', color='darkgreen')
    ax1.errorbar(bin_centres, data, yerr=data_err, fmt='ok',label='data')
#plt.semilogy()
    ax1.legend()
    ax1.set_ylim(1e-4,ax1.get_ylim()[1])
    if data_driven:
        ax1.set_title('ST_mult '+label+' QCD MC and real data, binned from data')
    else:
        ax1.set_title('ST_mult '+label+' QCD MC and real data, binned from MC')
    if mode in ['signal_search','signal_search_inj']:
        ratio = data/(mc+signal)
        ratio_err = data_err/(mc+signal)
    else:
        ratio = data/mc
        ratio_err = data_err/mc
    fill_between_steps(ax2, bin_edges, ratio+ratio_err ,ratio-ratio_err, alpha=0.2, step_where='pre',linewidth=0,color='red')
    ax2.errorbar(bin_centres, ratio, yerr=None, xerr=[np.abs(bin_edges[0:-1]-bin_centres),np.abs(bin_edges[1:]-bin_centres)], fmt='ok')
    ax2.set_xlabel('ST (GeV)',fontsize=17)
    ax2.set_ylabel('Data/MC',fontsize=17)
    ax1.set_ylabel(r'N/$\Delta$x',fontsize=17)
    ylims=[0.1,2]
    #ylims = ax2.get_ylim()
    #if ylims[0]>1: ylims[0] = 0.995
    #if ylims[1]<1: ylims[1] = 1.005
    ax2.set_ylim(ylims[0],ylims[1])
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.axhline(1,linewidth=2,color='r')
    tickbins = len(ax1.get_yticklabels()) # added
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='upper'))
    if suffix: suffix = '_'.join([suffix,mode])
    else: suffix = mode

    if data_driven:
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_databin'
        if suffix: save_name+='_'+suffix
        save_name+='.pdf'
    else:
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_mcbin'
        if suffix: save_name+='_'+suffix
        save_name+='.pdf'
    plt.savefig(save_name)
