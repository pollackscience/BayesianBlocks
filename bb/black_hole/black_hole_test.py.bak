#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from astroML.plotting import hist
from bayesian_blocks_modified import bayesian_blocks
import scipy.stats as st
from fill_between_steps import fill_between_steps
import cPickle as pkl

import ROOT
from ROOT import TFile
from ROOT import gDirectory
from ROOT import TF1
from ROOT import TRandom3

plt.close('all')
normed = True
log = True
mult = 'ST_mul5'
#mult = 'ST_mul2'
ROOT.gRandom.SetSeed(8675309)

ST_dict_data = pkl.load(open( "files/BHTree.p", "rb" ))
print 'pickle loaded'

ST_dict_data[mult]=[i for i in  ST_dict_data[mult] if i>2300]
nentries = len(ST_dict_data[mult])
print nentries

my_pdf = TF1("my_pdf","2.3949e6*3.74/(0.67472+x)**10.1809",2300,6000)
#my_pdf = TF1("my_pdf","2.3949e6/(0.67472+x)**10.1809",1900,5500)
my_rands = []
#for i in xrange(nentries):
for i in xrange(nentries):
    my_rands.append(my_pdf.GetRandom())

normed_counts_mc, bb_edges = np.histogram(my_rands,bayesian_blocks(my_rands), density=True)
normed_counts_data, _= np.histogram(ST_dict_data[mult],bb_edges, density=True)
counts_mc, _= np.histogram(my_rands,bb_edges)
counts_data, _= np.histogram(ST_dict_data[mult],bb_edges)

rescaled_counts_mc = normed_counts_mc*nentries
rescaled_counts_data = normed_counts_data*nentries
bin_centres = (bb_edges[:-1] + bb_edges[1:])/2.

rescaled_err = np.sqrt(counts_data)/(bb_edges[1:]-bb_edges[:-1])
err = np.sqrt(counts_data)

fig = plt.figure()
ax = plt.gca()
ax.set_yscale("log", nonposy='clip')
fill_between_steps(ax, bb_edges, rescaled_counts_mc,1e-3, alpha=0.2, step_where='pre',linewidth=0,label='fit MC')
ax.errorbar(bin_centres, rescaled_counts_data, yerr=rescaled_err, fmt='ok',label='data')
#plt.semilogy()
ax.legend()
plt.title('MC gen from '+mult+' fit function and real data')
plt.xlabel('ST (GeV)')
plt.ylabel(r'N/$\Delta$x')
plt.show()
plt.savefig('plots/'+mult+'_fit_and_data_normed.pdf')

fig = plt.figure()
ax = plt.gca()
ax.set_yscale("log", nonposy='clip')
fill_between_steps(ax, bb_edges, counts_mc,1e-3, alpha=0.2, step_where='pre',linewidth=0,label='fit MC')
ax.errorbar(bin_centres, counts_data, yerr=err, fmt='ok',label='data')
#plt.semilogy()
ax.legend()
plt.title('MC gen from '+mult+' fit function and real data')
plt.xlabel('ST (GeV)')
plt.ylabel('N')
plt.show()
plt.savefig('plots/'+mult+'_fit_and_data.pdf')

