#! /usr/bin/env python

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from hist_tools_modified import hist
from fill_between_steps import fill_between_steps
from matplotlib.ticker import MaxNLocator

from astroML.density_estimation import\
    scotts_bin_width, freedman_bin_width,\
    knuth_bin_width
#from bb_poly import bayesian_blocks
from bayesian_blocks_modified import bayesian_blocks

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
import plotly.graph_objs as go
import plotly.plotly as py

def make_hist_ratio_blackhole(bin_edges, data, mc, data_err, label, suffix = None, bg_est='data_driven', signal=None, mode='no_signal'):
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
    if bg_est in ['data_driven','mc']:
        fill_between_steps(ax1, bin_edges, mc,1e-4, alpha=0.2, step_where='pre',linewidth=0,label='QCD MC')
    else:
        fill_between_steps(ax1, bin_edges, mc,1e-4, alpha=0.2, step_where='pre',linewidth=0,label='ST_mul2 excl. (normed)')
    if mode in ['signal_search','signal_search_inj']:
        fill_between_steps(ax1, bin_edges,mc+signal,mc,alpha=0.6,step_where='pre',linewidth=0,label='Signal', color='darkgreen')
    ax1.errorbar(bin_centres, data, yerr=data_err, fmt='ok',label='data')
#plt.semilogy()
    ax1.legend()
    ax1.set_ylim(1e-4,ax1.get_ylim()[1])
    if bg_est=='data_driven':
        ax1.set_title('ST_mult '+label+' QCD MC and real data, binned from data')
    elif bg_est=='mc':
        ax1.set_title('ST_mult '+label+' QCD MC and real data, binned from MC')
    elif bg_est=='low_ST':
        ax1.set_title('ST_mult '+label+' data, bg est from ST mult_2 data')
    if mode in ['signal_search','signal_search_inj']:
        ratio = data/(mc+signal)
        ratio_err = data_err/(mc+signal)
    else:
        ratio = data/mc
        ratio_err = data_err/mc
    fill_between_steps(ax2, bin_edges, ratio+ratio_err ,ratio-ratio_err, alpha=0.2, step_where='pre',linewidth=0,color='red')
    ax2.errorbar(bin_centres, ratio, yerr=None, xerr=[np.abs(bin_edges[0:-1]-bin_centres),np.abs(bin_edges[1:]-bin_centres)], fmt='ok')
    ax2.set_xlabel('ST (GeV)',fontsize=17)
    ax2.set_ylabel('Data/BG',fontsize=17)
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

    if bg_est=='data_driven':
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_databin'
    elif bg_est=='mc':
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_mcbin'
    else:
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_st2_bg'

    if suffix: save_name+='_'+suffix
    save_name+='.pdf'
    plt.savefig(save_name)

def make_hist_ratio_blackhole2(bin_edges, data, mc, data_err, label, suffix = None, bg_est='data_driven', signal=None, mode='no_signal'):
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
    if bg_est in ['data_driven','mc']:
        #fill_between_steps(ax1, bin_edges, mc,1e-4, alpha=0.2, step_where='pre',linewidth=0,label='QCD MC')
        hist(np.asarray([mc,signal]).T,bin_edges, ax=ax1, alpha=0.2)
    else:
        fill_between_steps(ax1, bin_edges, mc,1e-4, alpha=0.2, step_where='pre',linewidth=0,label='ST_mul2 excl. (normed)')
    if mode in ['signal_search','signal_search_inj']:
        fill_between_steps(ax1, bin_edges,mc+signal,mc,alpha=0.6,step_where='pre',linewidth=0,label='Signal', color='darkgreen')
    ax1.errorbar(bin_centres, data, yerr=data_err, fmt='ok',label='data')
#plt.semilogy()
    ax1.legend()
    ax1.set_ylim(1e-4,ax1.get_ylim()[1])
    if bg_est=='data_driven':
        ax1.set_title('ST_mult '+label+' QCD MC and real data, binned from data')
    elif bg_est=='mc':
        ax1.set_title('ST_mult '+label+' QCD MC and real data, binned from MC')
    elif bg_est=='low_ST':
        ax1.set_title('ST_mult '+label+' data, bg est from ST mult_2 data')
    if mode in ['signal_search','signal_search_inj']:
        ratio = data/(mc+signal)
        ratio_err = data_err/(mc+signal)
    else:
        ratio = data/mc
        ratio_err = data_err/mc
    fill_between_steps(ax2, bin_edges, ratio+ratio_err ,ratio-ratio_err, alpha=0.2, step_where='pre',linewidth=0,color='red')
    ax2.errorbar(bin_centres, ratio, yerr=None, xerr=[np.abs(bin_edges[0:-1]-bin_centres),np.abs(bin_edges[1:]-bin_centres)], fmt='ok')
    ax2.set_xlabel('ST (GeV)',fontsize=17)
    ax2.set_ylabel('Data/BG',fontsize=17)
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

    if bg_est=='data_driven':
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_databin'
    elif bg_est=='mc':
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_mcbin'
    else:
        save_name = '../../plots/ST_mul'+label+'_mc_and_data_normed_st2_bg'

    if suffix: save_name+='_'+suffix
    save_name+='.pdf'
    plt.savefig(save_name)


def make_comp_plots(data, p0, save_dir,title='Plot of thing vs thing', xlabel='X axis', ylabel='Y axis',save_name='plot'):
    bb_edges = bayesian_blocks(data,p0=p0)
    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,bins=100,histtype='stepfilled',alpha=0.2,label='100 bins',normed=True)
    hist(data,bins=bb_edges,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_binsVbb.pdf')

    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,'knuth',histtype='stepfilled',alpha=0.2,label='knuth',normed=True)
    hist(data,bins=bb_edges,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_knuthVbb.pdf')

    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,'scott',histtype='stepfilled',alpha=0.2,label='scott',normed=True)
    hist(data,bins=bb_edges,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_scottVbb.pdf')


    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,'freedman',histtype='stepfilled',alpha=0.2,label='freedman',normed=True)
    hist(data,bins=bb_edges,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_freedmanVbb.pdf')

    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,bins=bb_edges,histtype='stepfilled',alpha=0.4,label='b blocks',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_bb.pdf')

    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,bins=100,histtype='stepfilled',alpha=0.4,label='100 bins',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_bins.pdf')

    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,bins='knuth',histtype='stepfilled',alpha=0.4,label='knuth',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_knuth.pdf')

    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,bins='scott',histtype='stepfilled',alpha=0.4,label='scott',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_scott.pdf')

    plt.figure()
    plt.yscale('log', nonposy='clip')
    hist(data,bins='freedman',histtype='stepfilled',alpha=0.4,label='freedman',normed=True)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_freedman.pdf')

###################################################################
# Plotly currently does not have variable bin width functionality #
###################################################################

#def make_comp_plots_plotly(data, p0, save_dir='/Users/brianpollack/Documents/PersonalWebPage/bb_plots/',title='Plot of thing vs thing', xlabel='X axis', ylabel='Y axis',save_name='plot'):
#    layout = go.Layout(
#                    title = title,
#                    autosize=False,
#                    width=675,
#                    height=650,
#                    xaxis=dict(title=xlabel),
#                    yaxis=dict(title=ylabel,type='log'),
#                    )
#    bb_edges = bayesian_blocks(data,p0=p0)
#    fig = plt.figure()
#    plt.yscale('log', nonposy='clip')
#    hist(data,bins=100,histtype='stepfilled',alpha=0.2,label='100 bins',normed=True)
#    hist(data,bins=bb_edges,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=True)
#    plt.legend()
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.title(title)
#    py.plot_mpl(fig,save_dir+save_name+'.html')
    #plt.savefig(save_dir+save_name+'_binsVbb.pdf')

    #uniform_hist = go.Histogram(x=data,name='Uniform bin size', histnorm='count')
    #bb_hist = go.Histogram(x=data,name='Bayesian Blocks',)
    #plotly_hists = [bins_regular]
    #fig = go.Figure(data=plotly_hists,layout=layout)
    #plot_html = new_iplot(fig,show_link=False)
    #plot(fig,filename = save_dir+save_name+'.html')

def make_bb_plot(data, p0, save_dir, range=None,title='Plot of thing vs thing', xlabel='X axis', ylabel='Y axis',save_name='plot', overlay_reg_bins = True, edges=None,scale=None, bins=80):

    normed=False
    if scale=='normed':
        normed=True
        scale=None

    if edges != None:
        bb_edges=edges
    else:
        bb_edges = bayesian_blocks(data,p0=p0)
    plt.figure()
    #bin_content = np.histogram(data,bb_edges,density=True)[0]
    #plt.yscale('log', nonposy='clip')

    hist(data,bins=bins,range=range,histtype='stepfilled',alpha=0.2,label='{} bins'.format(bins),normed=normed,scale=scale)
    #hist(data,bins=100,histtype='stepfilled',alpha=0.2,label='100 bins',normed=False)
    bb_content, bb_edges,_ = hist(data,bins=bb_edges,range=range,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=normed,scale=scale)
    #fill_between_steps(plt.gca(), bb_edges, bin_content*len(data),bin_content*len(data)/2, alpha=0.5, step_where='pre',linewidth=2,label='norm attempt')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_bb.pdf')
    return bb_content,bb_edges


def make_bb_plot_v2(data, p0, save_dir, hrange=None,title='Plot of thing vs thing', xlabel='X axis',
                    ylabel='Y axis', save_name='plot', edges=None,scale=None, bin_width=100,
                    logy=False, color='blue'):

    normed=False
    if scale=='normed':
        normed=True
        scale=None

    if edges != None:
        bb_edges=edges
    else:
        bb_edges = bayesian_blocks(data,p0=p0)

    if hrange == None:
        hrange=(min(data), max(data))

    nbins = (hrange[1]-hrange[0])/bin_width
    uni_edges = np.linspace(hrange[0], hrange[1], nbins+1)
    plt.figure()
    #bin_content = np.histogram(data,bb_edges,density=True)[0]
    if logy:
        plt.yscale('log', nonposy='clip')


    hist(data, bins=uni_edges,range=hrange,histtype='stepfilled',alpha=0.2,
         label='{} GeV bins'.format(bin_width),normed=normed,scale=scale, color=color)
    #hist(data,bins=100,histtype='stepfilled',alpha=0.2,label='100 bins',normed=False)
    bb_content, bb_edges,_ = hist(data,bins=bb_edges,range=hrange,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=normed,scale=scale)
    #fill_between_steps(plt.gca(), bb_edges, bin_content*len(data),bin_content*len(data)/2, alpha=0.5, step_where='pre',linewidth=2,label='norm attempt')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_dir+save_name+'_bb.pdf')
    return bb_content,bb_edges


def make_fit_plot(data, bins, range, frozen_pdf, title, xlabel='M (GeV)', ylabel='Count',
                  hist_label = 'hist', pdf_label = 'pdf', extra_pdf_tuple=None, textstr=None, ax=None):
    x       = np.linspace(range[0], range[1], 10000)
    binning = (range[1]-range[0])/bins
    if not ax:
        plt.figure()
        ax = plt.subplot()
    plt.hist(data, bins, range=range, alpha=0.2, histtype='stepfilled', label=hist_label)
    plt.plot(x, (len(data)*binning)*frozen_pdf(x), linewidth=2, label=pdf_label)
    if extra_pdf_tuple != None:
        '''extra_pdf_tuple = (extra_frozen_pdf, extra_scale, extra_label)'''
        plt.plot(x, (len(data)*binning*extra_pdf_tuple[1])*extra_pdf_tuple[0](x), 'k--', label=extra_pdf_tuple[2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if textstr:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # plt.text(0.85, 0.8, textstr, transform=plt.gca().transAxes, fontsize=14,
        #     verticalalignment='top', bbox=props)
        plt.text(0.86, 0.7, textstr, transform=plt.gca().transAxes, fontsize=16,
            verticalalignment='top', bbox=props)
    return ax
