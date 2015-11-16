#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from hist_tools_modified import hist
from bayesian_blocks_modified import bayesian_blocks
import re
import cPickle as pkl
import scipy.stats as st
from fill_between_steps import fill_between_steps
import cPickle as pkl

from ROOT import TFile
from ROOT import gDirectory

plt.close('all')
normed = True
log = True
ST_low = 2300
samples = 500
seed = 10
p0=0.01

df_mc = pkl.load(open('files/BHTree_mc.p','rb'))
df_data = pkl.load(open('files/BHTree_data.p','rb'))

#weights: array([ 0.27436519,  0.04019762,  0.01657276])
df_mc1 = df_mc[np.isclose(df_mc.weightTree,0.27436519)]
df_mc2 = df_mc[np.isclose(df_mc.weightTree, 0.0401976)]
df_mc3 = df_mc[np.isclose(df_mc.weightTree, 0.01657276)]

for ST in range(2,11):
    #my_ST = df_mc1[df_mc1['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].sample(samples*42,random_state=seed,replace=False).values
    my_ST_data = df_data[df_data['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].values
    nentries = len(my_ST_data)
    df_mc2_st = df_mc2[df_mc2['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)]
    df_mc3_st = df_mc3[df_mc3['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)]
    props = [len(df_mc2_st)* 0.040197, len(df_mc3_st)*0.01657276]
    props = [i/min(props) for i in props]
    props = [i/sum(props) for i in props]
    print props
    my_ST_mc = []
    my_ST_mc = np.append(my_ST_mc, df_mc2_st.sample(int(nentries*props[0]),random_state=seed).values)
    my_ST_mc = np.append(my_ST_mc, df_mc3_st.sample(int(nentries*props[1]),random_state=seed).values)
    print 'n_data',nentries
    print 'n_mc',len(my_ST_mc)

    normed_counts_mc, bb_edges = np.histogram(my_ST_mc,bayesian_blocks(my_ST_mc,p0=p0), density=True)
    normed_counts_data, _= np.histogram(my_ST_data,bb_edges, density=True)
    counts_mc, _= np.histogram(my_ST_mc,bb_edges)
    counts_data, _= np.histogram(my_ST_data,bb_edges)

    rescaled_counts_mc = normed_counts_mc*nentries
    rescaled_counts_data = normed_counts_data*nentries
    bin_centres = (bb_edges[:-1] + bb_edges[1:])/2.

    rescaled_err = np.sqrt(counts_data)/(bb_edges[1:]-bb_edges[:-1])
    err = np.sqrt(counts_data)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')
    fill_between_steps(ax, bb_edges, rescaled_counts_mc,1e-4, alpha=0.2, step_where='pre',linewidth=0,label='QCD MC')
    ax.errorbar(bin_centres, rescaled_counts_data, yerr=rescaled_err, fmt='ok',label='data')
#plt.semilogy()
    ax.legend()
    plt.title('ST_mult '+str(ST)+' QCD MC and real data')
    plt.xlabel('ST (GeV)')
    plt.ylabel(r'N/$\Delta$x')
    plt.show()
    plt.savefig('plots/ST_mul'+str(ST)+'_mc_and_data_normed.pdf')

