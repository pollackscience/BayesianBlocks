#! /usr/bin/env python

from __future__ import division
import numpy as np
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import cPickle as pkl
from bb.tools.hist_tools_modified import hist
from bb.tools.fill_between_steps import fill_between_steps
import os

plt.close('all')
current_dir = os.path.dirname(__file__)
bb_dir=os.path.join(current_dir,'../..')
#if not os.path.isfile(bb_dir+'/files/values2.p'):
#    z_data = np.loadtxt(bb_dir+'/files/values2.dat')
#    pkl.dump( z_data, open( bb_dir+'/files/values2.p', "wb" ),pkl.HIGHEST_PROTOCOL )
#else:
#    z_data = pkl.load(open(bb_dir+'/files/values2.p',"rb"))
print 'loaded'
#print z_data[0:20]

z_data = pkl.load(open(bb_dir+'/files/DY/ZLL_v2.p',"rb"))
z_data = z_data.query('50<Mll<150')
n_events = 10000

#be = bayesian_blocks(z_data[0:n_events].Muon1_Pt, p0=0.01)
be = bayesian_blocks(z_data[0:n_events].Mll, p0=0.01)
#be = np.linspace(50,151,25)
bin_centers = (be[1:]+be[:-1])/2

#make_comp_plots(z_data[0:50000], 0.01, bb_dir+'/plots/',title=r'Z$\to\mu\mu$ Data', xlabel=r'$m_{\ell\ell}$ (GeV)', ylabel='A.U.',save_name='z_data_hist')

#bc, be = make_bb_plot(z_data[0:100000].Muon1_Pt, 0.01, bb_dir+'/plots/', title=r'$Z\rightarrow\mu\mu$',
#        xlabel=r'$pT_{\mu}$ (GeV)', ylabel='Count/(bin width)',save_name='ZLL_muon', bins=100, scale='binwidth')
#plt.show()
xlims = (50,150)

fig = plt.figure()
gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
ax1=fig.add_subplot(gs[0])
ax1.set_yscale("log", nonposy='clip')
ax2=fig.add_subplot(gs[1],sharex=ax1)
ax1.grid(True)
ax2.grid(True)
plt.setp(ax1.get_xticklabels(), visible=False)
fig.subplots_adjust(hspace=0.001)
ax1.set_xlim(xlims)

bc_d, _, _  = hist(z_data[0:n_events].Mll, ax = ax1, bins = be, scale='binwidth',
        histtype='marker',markersize=10,color='k', errorbars=True, label='Reco')
bc_mc, _, _ = hist(z_data.Mll_gen, ax = ax1, bins = be, scale='binwidth',
        weights = [n_events/len(z_data)]*len(z_data), histtype = 'stepfilled', alpha=0.2, label='Gen')
ax1.legend()
ratio = bc_d/bc_mc
ratio_err = np.sqrt(bc_d)/bc_mc
fill_between_steps(ax2, be, ratio+ratio_err ,ratio-ratio_err, alpha=0.2, step_where='pre',linewidth=0,color='red')
ax2.errorbar(bin_centers, ratio, yerr=None, xerr=[np.abs(be[0:-1]-bin_centers),np.abs(be[1:]-bin_centers)], fmt='ok')
ax2.set_xlabel(r'$\mu\mu_{M}$ (GeV)',fontsize=17)
ax2.set_ylabel('Data/BG',fontsize=17)
ax1.set_ylabel(r'N/$\Delta$x',fontsize=17)
ax1.set_title('Z Mass distribution, Bayesian Blocks')
ax2.get_yaxis().get_major_formatter().set_useOffset(False)
ax2.axhline(1,linewidth=2,color='r')
tickbins = len(ax1.get_yticklabels()) # added
ax2.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='upper'))
plt.savefig('figures/bb_Z_gen_reco.pdf')
plt.savefig('figures/bb_Z_gen_reco.png')

plt.show()


