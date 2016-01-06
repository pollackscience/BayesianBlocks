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
import pandas as pd

plt.close('all')
current_dir = os.path.dirname(__file__)
bb_dir=os.path.join(current_dir,'../..')
hgg_bg = pkl.load(open(bb_dir+'/files/hgg_bg.p',"rb"))
hgg_signal = pkl.load(open(bb_dir+'/files/hgg_signal.p',"rb"))

hgg_bg_sm_range = hgg_bg[(hgg_bg.Mgg>=100)&(hgg_bg.Mgg<=160)].Mgg
hgg_signal_selection = hgg_signal[(hgg_signal.Mgg>=120)&(hgg_signal.Mgg<=130)][0:200].Mgg

print 'loaded'
#print z_data[0:20]

make_comp_plots(hgg_bg_sm_range, 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_bg_hist')
make_comp_plots(hgg_signal_selection, 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_signal_hist')
make_comp_plots(pd.concat([hgg_bg_sm_range,hgg_signal_selection],ignore_index=True), 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_inject_hist')


