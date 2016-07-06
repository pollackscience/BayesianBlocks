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
from bb.tools.hist_tools_modified import hist
import os

plt.close('all')
p0=0.05

current_dir = os.path.dirname(__file__)
bb_dir=os.path.join(current_dir,'../..')
df_data = pkl.load(open(bb_dir+'/files/BH/BH_paper_data.p','rb'))

data_ST_mul8 = df_data[df_data.ST_mul8_BB>=2800].ST_mul8_BB.values

bc,be,_ = hist(data_ST_mul8, p0=p0, bins='blocks', scale='binwidth', log=True)
print be

plt.show()
