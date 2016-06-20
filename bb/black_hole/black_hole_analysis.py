#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from matplotlib import pyplot as plt
import cPickle as pkl
import scipy.stats as st
import cPickle as pkl
from bb.tools.bb_plotter import make_hist_ratio_blackhole
from bb.tools.bb_plotter import make_hist_ratio_blackhole2

def do_bh_analysis():

#set up variables
    plt.close('all')
    normed = True
    log = True
    STs = [2,3,4,5,6,7,8,9,10]
    ST_low = [2300,2300,2300,2600,2600,2600,2800,2800,2900]
    ST_low = [2300]*9
    ST_low_dict = dict(zip(STs,ST_low))
    samples = 5000
    seed = 2
    p0=0.005
    bg_est = 'data_driven' #'data_driven','mc','low_ST'
    mode = 'signal_search' #'no_signal','signal_search','signal_inj','signal_search_inj'

    if mode not in ['no_signal','signal_search','signal_inj','signal_search_inj']: raise KeyError('mode is not allowed!')
    if bg_est not in ['data_driven','mc','low_ST']: raise KeyError('bg_est is not allowed!')

    if mode in ['signal_search','signal_inj','signal_search_inj']:
        signal_num = 10
    else:
        signal_num = 0

    df_mc = pkl.load(open('../../files/BH/BHTree_mc.p','rb'))
    df_signal = pkl.load(open('../../files/BH/BHTree_signal.p','rb'))
    df_data = pkl.load(open('../../files/BH/BHTree_data.p','rb'))

    weights = df_mc.weightTree.unique()#[0.27436519,0.0401976,0.01657276]
    df_mc_list = []
    for weight in weights:
        df_mc_list.append(df_mc[np.isclose(df_mc.weightTree,weight)])

    all_edges = {}
    #for ST in range(2,11):
    for ST in [8]:
        my_ST_data = df_data[df_data['ST_mul'+str(ST)]>ST_low_dict[ST]]['ST_mul'+str(ST)].values
        nentries = len(my_ST_data)
        my_ST_mc = []
        if bg_est == 'low_ST':
            my_ST_mc = df_data[df_data['ST_mul2']>ST_low_dict[ST]][df_data['n_multiplicity']==2]['ST_mul2'].values
        else:
            df_mc_st_list = [df[df['ST_mul'+str(ST)]>ST_low_dict[ST]]['ST_mul'+str(ST)] for df in df_mc_list]
            if mode in ['signal_search','signal_inj','signal_search_inj']:
                my_ST_signal = df_signal[df_signal['ST_mul'+str(ST)]>ST_low_dict[ST]]['ST_mul'+str(ST)]

            samples,rel_weights = find_sample_number(df_mc_st_list,weights)
            for i,mc in enumerate(df_mc_st_list):
                if samples*rel_weights[i]==0: continue
                my_ST_mc = np.append(my_ST_mc, mc.sample(int(samples*rel_weights[i]),random_state=seed).values)

        print 'ST_mult',ST
        print '   n_data',nentries
        print '   n_mc',len(my_ST_mc)

#get the edges from bb, and the normalized bin values (integral of all hists is 1)
        #if signal and inject:
        #    my_ST_data = np.append(my_ST_data,my_ST_signal.

        if mode in ['signal_inj','signal_search_inj']:
            my_ST_data = np.append(my_ST_data, my_ST_signal.sample(signal_num,random_state=seed).values)
            nentries+=signal_num
        elif mode in ['signal_search']:
            my_ST_signal = my_ST_signal.values
        return my_ST_data, my_ST_mc, my_ST_signal

        print len(my_ST_data)
        normed_counts_data, bb_edges = np.histogram(my_ST_data,bayesian_blocks(my_ST_data,p0=p0), density=True)
        normed_counts_data_nobb, nobb_edges = np.histogram(my_ST_data,20, density=True)
        normed_counts_mc, _= np.histogram(my_ST_mc,bb_edges, density=True)
        normed_counts_mc_nobb, _= np.histogram(my_ST_mc,nobb_edges, density=True)
        if mode in ['signal_search','signal_search_inj']:
            normed_counts_signal, _= np.histogram(my_ST_signal,bb_edges, density=True)
            normed_counts_signal_nobb, _= np.histogram(my_ST_signal,nobb_edges, density=True)

#rescale the values so that the integral of the data hist is = num of entries
        rescaled_counts_data = normed_counts_data*nentries
        rescaled_counts_data_nobb = normed_counts_data_nobb*nentries
        if mode in ['signal_search','signal_search_inj']:
            rescaled_counts_mc = normed_counts_mc*(nentries-signal_num)
            rescaled_counts_mc_nobb = normed_counts_mc_nobb*(nentries-signal_num)
            rescaled_counts_signal = normed_counts_signal*signal_num
            rescaled_counts_signal_nobb = normed_counts_signal_nobb*signal_num
        else:
            rescaled_counts_mc = normed_counts_mc*(nentries)
            rescaled_counts_mc_nobb = normed_counts_mc_nobb*(nentries)

#properly calculate the error bars on the data
        counts_data, _= np.histogram(my_ST_data,bb_edges)
        counts_data_nobb, _= np.histogram(my_ST_data,nobb_edges)
        rescaled_err = np.sqrt(counts_data)/(bb_edges[1:]-bb_edges[:-1])
        rescaled_err_nobb = np.sqrt(counts_data_nobb)/(nobb_edges[1:]-nobb_edges[:-1])
        err = np.sqrt(counts_data)
#properly account for the BG error for ratio plot
        counts_bg, _= np.histogram(my_ST_mc,bb_edges)
        counts_bg_nobb, _= np.histogram(my_ST_mc,nobb_edges)
        rescaled_err_bg = np.sqrt(counts_bg)/(bb_edges[1:]-bb_edges[:-1])
        rescaled_err_bg_nobb = np.sqrt(counts_bg_nobb)/(nobb_edges[1:]-nobb_edges[:-1])

        if mode in ['signal_search','signal_search_inj']:
            make_hist_ratio_blackhole(bb_edges, rescaled_counts_data, rescaled_counts_mc, rescaled_err, str(ST), suffix = None, bg_est=bg_est, signal = rescaled_counts_signal, mode = mode)
            make_hist_ratio_blackhole2(nobb_edges, rescaled_counts_data_nobb, rescaled_counts_mc_nobb, rescaled_err_nobb, str(ST), suffix = 'nobb', bg_est=bg_est, signal = rescaled_counts_signal_nobb, mode=mode)
        else:
            make_hist_ratio_blackhole(bb_edges, rescaled_counts_data, rescaled_counts_mc, rescaled_err, str(ST), suffix = None, bg_est=bg_est, mode=mode)
            make_hist_ratio_blackhole(nobb_edges, rescaled_counts_data_nobb, rescaled_counts_mc_nobb, rescaled_err_nobb, str(ST), suffix = 'nobb', bg_est=bg_est, mode=mode)

        plt.show()

        all_edges[ST]=bb_edges

    for key in all_edges:
        print 'ST'+str(key), all_edges[key]
    return all_edges


def find_sample_number(df_list,weights):
    props = [len(df_list[i])*weights[i] for i in range(len(df_list))]
    props = [i/min(props) for i in props]
    props = [i/sum(props) for i in props]
    props = [np.nan_to_num(i) for i in props]
    for sample in range(min(map(len,df_list)), max(map(len,df_list))):
        for j in range(len(props)):
            if int(sample*props[j])>len(df_list[j]):
                return (sample-1,props)
    return (sample, props)

if __name__ =="__main__":
    data,mc,signal = do_bh_analysis()

