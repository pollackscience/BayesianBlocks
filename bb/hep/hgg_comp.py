#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from bb.tools.bayesian_blocks_modified import bayesian_blocks
from matplotlib import pyplot as plt
import cPickle as pkl
import scipy.stats as st
import cPickle as pkl
from bb.tools.bb_plotter import make_comp_plots, make_bb_plot
from bb.tools.hist_tools_modified import hist
import nllfitter.future_fitter as ff
from nllfitter.fitter import fit_plot
import os
import pandas as pd
from scipy.stats import norm
from ROOT import TF1
from ROOT import TRandom3

def scale_data(x, xlow=100., xhigh=180., invert=False):
    if not invert:
        return 2*(x - xlow)/(xhigh - xlow) - 1
    else:
        return 0.5*(x + 1)*(xhigh - xlow) + xlow

def bg_pdf(x,a,xlow=100,xhigh=180):
    '''3rd order legendre poly, mapped from [-1,1] to [xlow,xhigh].
    mapping: f(t) = -1+((1--1)/(xhigh-xlow))*(t-xlow)'''
    t = -1+((1+1)/(xhigh-xlow))*(x-xlow)
    return 0.5 + a[0]*t + 0.5*a[1]*(3*t**2 - 1) + 0.5*a[2]*(5*t**3 - 3*t)


plt.close('all')
current_dir = os.path.dirname(__file__)
bb_dir=os.path.join(current_dir,'../..')
hgg_bg = pkl.load(open(bb_dir+'/files/hgg_bg.p',"rb"))
hgg_signal = pkl.load(open(bb_dir+'/files/hgg_signal.p',"rb"))

hgg_bg_sm_range = hgg_bg[(hgg_bg.Mgg>100)&(hgg_bg.Mgg<180)].Mgg
#hgg_signal_selection = hgg_signal[(hgg_signal.Mgg>=120)&(hgg_signal.Mgg<=130)][0:500].Mgg
#data = scale_data(hgg_bg_sm_range.values)
data = hgg_bg_sm_range.values

print 'loaded'
#print z_data[0:20]

xlimits  = (100., 180.)
sdict    = {'mu'    : lambda x : scale_data(x, invert = True),
            'sigma' : lambda x : x*(xlimits[1] - xlimits[0])/2.,
           }

#make_bb_plot(hgg_bg_sm_range, 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_bg_hist')
#make_bb_plot(hgg_signal_selection, 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_signal_hist')
#edges = make_bb_plot(pd.concat([hgg_bg_sm_range,hgg_signal_selection],ignore_index=True), 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin',save_name='hgg_inject_hist')
#make_bb_plot(pd.concat([hgg_bg_sm_range,hgg_signal_selection],ignore_index=True), 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_inject_hist',)
#plt.show()

bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])
bg_fitter = ff.NLLFitter(bg_model, data)
bg_result = bg_fitter.fit([0.0, 0.0, 0.0])

print bg_result
#raw_input()
#t = -1+((1+1)/(xhigh-xlow))*(x-xlow)
#return 0.5 + a[0]*t + 0.5*a[1]*(3*t**2 - 1) + 0.5*a[2]*(5*t**3 - 3*t)
legendre_str = "(-1+((1+1)/({1}-{0}))*(x-{0}))".format(xlimits[0],xlimits[1])
legendre_str = "0.5 + {0}*{3} + 0.5*{1}*(3*{3}**2 - 1) + 0.5*{2}*(5*{3}**3 - 3*{3})".format(bg_result.x[0],bg_result.x[1],bg_result.x[2],legendre_str)

#raw_input()
my_pdf = TF1("my_pdf",legendre_str,xlimits[0],xlimits[1])
my_rands = []
for i in xrange(len(data)):
    my_rands.append(my_pdf.GetRandom())
#print my_rands
#sig_pdf = lambda x, a: (1 - a[0])*bg_pdf(x, a[3:5]) + a[0]*norm.pdf(x, a[1], a[2])
plt.hist(data)
plt.hist(my_rands)

### Define bg+sig model and carry out fit ###
#sig_model = ff.Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2'])
#sig_model.set_bounds([(0., 0),
#                  (0, 0), (0., 0.0),
#                  (0., 0.), (0., 0.)])
#sig_fitter = ff.NLLFitter(sig_model, data, scaledict=sdict)
#sig_result = sig_fitter.fit((0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]))

### Plots!!! ###
#print 'Making plot of fit results.'
#fit_plot(scale_data(data, invert=True), xlimits, None, None, bg_pdf, bg_result.x, 'test')
#fit_plot(data, xlimits, None, None, bg_pdf, bg_result.x, 'test')
plt.show()
