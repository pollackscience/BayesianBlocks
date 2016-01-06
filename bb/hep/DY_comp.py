#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from matplotlib import pyplot as plt
import cPickle as pkl
import scipy.stats as st
import cPickle as pkl
from bb.tools.bb_plotter import make_comp_plots
from bb.tools.hist_tools_modified import hist
import os

plt.close('all')
current_dir = os.path.dirname(__file__)
bb_dir=os.path.join(current_dir,'../..')
if not os.path.isfile(bb_dir+'/files/values2.p'):
    z_data = np.loadtxt(bb_dir+'/files/values2.dat')
    pkl.dump( z_data, open( bb_dir+'/files/values2.p', "wb" ),pkl.HIGHEST_PROTOCOL )
else:
    z_data = pkl.load(open(bb_dir+'/files/values2.p',"rb"))
print 'loaded'
#print z_data[0:20]

make_comp_plots(z_data[0:50000], 0.01, bb_dir+'/plots/',title=r'Z$\to\mu\mu$ Data', xlabel=r'$m_{\ell\ell}$ (GeV)', ylabel='A.U.',save_name='z_data_hist')
'''
plt.yscale('log', nonposy='clip')
hist(z_data[0:50000],'knuth',histtype='stepfilled',alpha=0.2,label='knuth',normed=True)
hist(z_data[0:50000],'scott',histtype='stepfilled',alpha=0.2,label='scott',normed=True)
hist(z_data[0:50000],'freedman',histtype='stepfilled',alpha=0.2,label='freedman',normed=True)
hist(z_data[0:50000],'blocks',fitness='events',p0=0.01,histtype='step',linewidth=2.0,color='crimson',label='b blocks',normed=True)
plt.legend()
plt.xlabel(r'$m_{\ell\ell}$ (GeV)')
plt.ylabel('A.U.')
plt.title(r'Z$\to\mu\mu$ Data')
plt.savefig(bb_dir+'/plots/z_data_hist_comp.pdf')

plt.figure()
plt.yscale('log', nonposy='clip')
hist(z_data[0:50000],'knuth',histtype='stepfilled',alpha=0.4,label='knuth',normed=True)
plt.legend()
plt.xlabel(r'$m_{\ell\ell}$ (GeV)')
plt.ylabel('A.U.')
plt.title(r'Z$\to\mu\mu$ Data')
plt.savefig(bb_dir+'/plots/z_data_hist_knuth.pdf')

plt.figure()
plt.yscale('log', nonposy='clip')
hist(z_data[0:50000],'scott',histtype='stepfilled',alpha=0.4,label='scott',normed=True)
plt.legend()
plt.xlabel(r'$m_{\ell\ell}$ (GeV)')
plt.ylabel('A.U.')
plt.title(r'Z$\to\mu\mu$ Data')
plt.savefig(bb_dir+'/plots/z_data_hist_scott.pdf')

plt.figure()
plt.yscale('log', nonposy='clip')
hist(z_data[0:50000],'freedman',histtype='stepfilled',alpha=0.4,label='freedman',normed=True)
plt.legend()
plt.xlabel(r'$m_{\ell\ell}$ (gev)')
plt.ylabel('a.u.')
plt.title(r'z$\to\mu\mu$ data')
plt.savefig(bb_dir+'/plots/z_data_hist_freedman.pdf')

plt.figure()
plt.yscale('log', nonposy='clip')
hist(z_data[0:50000],'blocks',fitness='events',p0=0.01,histtype='stepfilled', alpha=0.4, label='b blocks',normed=True)
plt.legend()
plt.xlabel(r'$m_{\ell\ell}$ (gev)')
plt.ylabel('a.u.')
plt.title(r'z$\to\mu\mu$ data')
plt.savefig(bb_dir+'/plots/z_data_hist_bb.pdf')
'''

plt.show()

