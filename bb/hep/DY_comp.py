#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from matplotlib import pyplot as plt
import cPickle as pkl
import scipy.stats as st
import cPickle as pkl
from bb.tools.bb_plotter import make_bb_plot
from bb.tools.hist_tools_modified import hist
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

z_data = pkl.load(open(bb_dir+'/files/DY/ZLL.p',"rb"))

#make_comp_plots(z_data[0:50000], 0.01, bb_dir+'/plots/',title=r'Z$\to\mu\mu$ Data', xlabel=r'$m_{\ell\ell}$ (GeV)', ylabel='A.U.',save_name='z_data_hist')

z_data = z_data.query('50<Mll<150')
bc, be = make_bb_plot(z_data[0:100000].Mll, 0.01, bb_dir+'/plots/', title=r'Z\rightarrow\mu\mu',
        xlabel=r'$m_{\ell\ell}$ (GeV)', ylabel='Prob/(bin width)',save_name='ZLL', bins=100)
plt.show()

