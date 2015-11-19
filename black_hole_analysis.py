#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from bayesian_blocks_modified import bayesian_blocks
from matplotlib import pyplot as plt
import cPickle as pkl
import scipy.stats as st
import cPickle as pkl
from bb_plotter import make_hist_ratio_blackhole

plt.close('all')
normed = True
log = True
ST_low = 2300
samples = 5000
seed = 2
p0=0.005
data_driven = True
signal = True
signal_num = 10

df_mc = pkl.load(open('files/BHTree_mc.p','rb'))
df_signal = pkl.load(open('files/BHTree_signal.p','rb'))
df_data = pkl.load(open('files/BHTree_data.p','rb'))

#weights: array([ 0.27436519,  0.04019762,  0.01657276])
df_mc1 = df_mc[np.isclose(df_mc.weightTree,0.27436519)]
df_mc2 = df_mc[np.isclose(df_mc.weightTree, 0.0401976)]
df_mc3 = df_mc[np.isclose(df_mc.weightTree, 0.01657276)]
weights = [0.27436519,0.0401976,0.01657276]
def find_sample_number(df_list,props):
    for i in range(min(map(len,df_list)), max(map(len,df_list))):
        for j in range(len(props)):
            if int(i*props[j])>len(df_list[j]):
                return i-1
    return i


all_edges = []
#for ST in range(2,11):
for ST in [10]:
    #my_ST = df_mc1[df_mc1['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].sample(samples*42,random_state=seed,replace=False).values
    my_ST_data = df_data[df_data['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].values
    nentries = len(my_ST_data)
    df_mc_st_list = []
    df_mc_st_list.append(df_mc1[df_mc1['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)])
    df_mc_st_list.append(df_mc2[df_mc2['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)])
    df_mc_st_list.append(df_mc3[df_mc3['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)])
    if signal:
        my_ST_signal = df_signal[df_signal['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].values
    #props = [len(df_mc2_st)* 0.040197, len(df_mc3_st)*0.01657276]
    props = [len(df_mc_st_list[i])*weights[i] for i in range(len(df_mc_st_list))]
    props = [i/min(props) for i in props]
    props = [i/sum(props) for i in props]
    print props
    print map(len,df_mc_st_list)
    my_ST_mc = []
    if data_driven:
        samples = find_sample_number(df_mc_st_list,props)
        #samples = min(len(df_mc1_st)*props[0], len(df_mc2_st)*props[1], len(df_mc3_st)*props[2])
        my_ST_mc = np.append(my_ST_mc, df_mc_st_list[0].sample(int(samples*props[0]),random_state=seed).values)
        my_ST_mc = np.append(my_ST_mc, df_mc_st_list[1].sample(int(samples*props[1]),random_state=seed).values)
        my_ST_mc = np.append(my_ST_mc, df_mc_st_list[2].sample(int(samples*props[2]),random_state=seed).values)
    else:
        my_ST_mc = np.append(my_ST_mc, df_mc2_st.sample(int(nentries*props[0]),random_state=seed).values)
        my_ST_mc = np.append(my_ST_mc, df_mc3_st.sample(int(nentries*props[1]),random_state=seed).values)

    print 'n_data',nentries
    print 'n_mc',len(my_ST_mc)

    if data_driven:
        normed_counts_data, bb_edges = np.histogram(my_ST_data,bayesian_blocks(my_ST_data,p0=p0), density=True)
        normed_counts_mc, _= np.histogram(my_ST_mc,bb_edges, density=True)

        normed_counts_data_nobb, nobb_edges = np.histogram(my_ST_data,20, density=True)
        normed_counts_mc_nobb, _= np.histogram(my_ST_mc,nobb_edges, density=True)
        if signal:
            normed_counts_signal, _= np.histogram(my_ST_signal,bb_edges, density=True)
            normed_counts_signal_nobb, _= np.histogram(my_ST_signal,nobb_edges, density=True)

    else:
        normed_counts_mc, bb_edges = np.histogram(my_ST_mc,bayesian_blocks(my_ST_mc,p0=p0), density=True)
        normed_counts_data, _= np.histogram(my_ST_data,bb_edges, density=True)


    rescaled_counts_data = normed_counts_data*nentries
    rescaled_counts_data_nobb = normed_counts_data_nobb*nentries

    if signal:
        rescaled_counts_mc = normed_counts_mc*(nentries-signal_num)
        rescaled_counts_mc_nobb = normed_counts_mc_nobb*(nentries-signal_num)

        rescaled_counts_signal = normed_counts_signal*signal_num
        rescaled_counts_signal_nobb = normed_counts_signal_nobb*signal_num


    counts_data, _= np.histogram(my_ST_data,bb_edges)
    counts_data_nobb, _= np.histogram(my_ST_data,nobb_edges)
    rescaled_err = np.sqrt(counts_data)/(bb_edges[1:]-bb_edges[:-1])
    rescaled_err_nobb = np.sqrt(counts_data_nobb)/(nobb_edges[1:]-nobb_edges[:-1])
    err = np.sqrt(counts_data)

    if signal:
        make_hist_ratio_blackhole(bb_edges, rescaled_counts_data, rescaled_counts_mc, rescaled_err, str(ST), suffix = None, data_driven=data_driven, signal = rescaled_counts_signal)
        make_hist_ratio_blackhole(nobb_edges, rescaled_counts_data_nobb, rescaled_counts_mc_nobb, rescaled_err_nobb, str(ST), suffix = 'nobb', data_driven=data_driven, signal = rescaled_counts_signal_nobb)
    else:
        make_hist_ratio_blackhole(bb_edges, rescaled_counts_data, rescaled_counts_mc, rescaled_err, str(ST), suffix = None, data_driven=data_driven)
        make_hist_ratio_blackhole(nobb_edges, rescaled_counts_data_nobb, rescaled_counts_mc_nobb, rescaled_err_nobb, str(ST), suffix = 'nobb', data_driven=data_driven)
    plt.show()

    all_edges.append(bb_edges)

for i,edges in enumerate(all_edges):
    print 'ST{}'.format(i+2)
    print repr(edges)

