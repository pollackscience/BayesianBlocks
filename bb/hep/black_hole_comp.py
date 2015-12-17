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
p0=0.005

current_dir = os.path.dirname(__file__)
bb_dir=os.path.join(current_dir,'../..')
df_data = pkl.load(open(bb_dir+'/files/BHTree_data.p','rb'))

my_ST_data = df_data[df_data['ST_mul5'>2300]['ST_mul8'].values

