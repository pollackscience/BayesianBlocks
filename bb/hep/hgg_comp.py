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
import functools

def bg_pdf(x,a,xlow=100,xhigh=180,doROOT=False):
    '''3rd order legendre poly, mapped from [-1,1] to [xlow,xhigh].
    mapping: f(t) = -1+((1--1)/(xhigh-xlow))*(t-xlow)
    a[0]: O(x)   free parameter
    a[1]: O(x^2) free parameter
    a[2]: O(x^3) free parameter
    Normalization factor calculated based on x-range
    '''
    if doROOT: x=x[0] # this is a ROOT-compatible hack
    t = -1+((1+1)/(xhigh-xlow))*(x-xlow)
    return 1/(xhigh-xlow)*(1.0 + a[0]*t + 0.5*a[1]*(3*t**2 - 1) + 0.5*a[2]*(5*t**3 - 3*t))

def sig_pdf(x,a,doROOT=False):
    '''simple gaussian pdf.
    a[0]: mean
    a[1]: sigma
    '''
    if doROOT: x=x[0] # this is a ROOT-compatible hack
    return (1.0/(a[1]*np.sqrt(2*np.pi)))*np.exp(-(x-a[0])**2/(2*a[1]**2))

def bg_sig_pdf(x,a,xlow=100,xhigh=180,doROOT=False):
    '''legendre bg pdf and gaussian signal pdf, with a relative normalization factor.
    a[0]: normalization and signal strength parameter
    a[1]: signal mean
    a[2]: signal sigma
    a[3]: bg O(x)   free parameter
    a[4]: bg O(x^2) free parameter
    a[5]: bg O(x^3) free parameter
    '''
    if doROOT: x=x[0]    # this is a ROOT-compatible hack
    b = [a[3],a[4],a[5]] # this is a ROOT-compatible hack
    c = [a[1],a[2]]      # this is a ROOT-compatible hack
    return (1 - a[0])*bg_pdf(x, b,xlow=xlow,xhigh=xhigh) + a[0]*sig_pdf(x,c)


plt.close('all')
current_dir = os.path.dirname(__file__)
bb_dir=os.path.join(current_dir,'../..')
hgg_bg = pkl.load(open(bb_dir+'/files/hgg_bg.p',"rb"))
hgg_signal = pkl.load(open(bb_dir+'/files/hgg_signal.p',"rb"))

# grab a handful of bg events, and an ~X sigma number of signal events
n_sigma              = 5
hgg_bg_selection     = hgg_bg[(hgg_bg.Mgg>100)&(hgg_bg.Mgg<180)][0:10000].Mgg
n_bg_under_sig       = hgg_bg_selection[(118<hgg_bg_selection)&(hgg_bg_selection<133)].size
n_sig                = int(n_sigma*np.sqrt(n_bg_under_sig))
hgg_signal_selection = hgg_signal[(hgg_signal.Mgg>=100)&(hgg_signal.Mgg<=180)][0:n_sig].Mgg
data_bg              = hgg_bg_selection.values
data_sig             = hgg_signal_selection.values
data_bg_sig          = np.concatenate((data_bg,data_sig))

print 'loaded'

# setup initial ranges and binning
xlimits = (100., 180.)
x       = np.linspace(xlimits[0],xlimits[1],10000)
nbins   = 80
binning = (xlimits[1]-xlimits[0])/nbins

# fit to the data distributions
bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])
bg_fitter = ff.NLLFitter(bg_model, data_bg)
bg_result = bg_fitter.fit([0.0, 0.0, 0.0])

sig_model = ff.Model(sig_pdf, ['mu','sigma'])
sig_model.set_bounds([(110, 130), (1, 5)])
sig_fitter = ff.NLLFitter(sig_model, data_sig)
sig_result = sig_fitter.fit([120.0, 2])

bg_sig_model = ff.Model(bg_sig_pdf, ['C','mu','sigma','a1', 'a2', 'a3'])
bg_sig_model.set_bounds([(0,1),(xlimits[0], xlimits[1]), (0, 50),(-1., 1.), (-1., 1.), (-1., 1.)])
bg_sig_fitter = ff.NLLFitter(bg_sig_model, data_bg_sig)
bg_sig_result = bg_sig_fitter.fit([0.01,125.0, 2,0.0, 0.0, 0.0])

# make MC
bg_sig_pdf_ROOT = functools.partial(bg_sig_pdf,doROOT=True)
tf1_bg_sig_pdf = TF1("tf1_bg_sig_pdf",bg_sig_pdf_ROOT,100,180,6)
tf1_bg_sig_pdf.SetParameters(*bg_sig_result.x)
mc_bg_sig = []
for i in xrange(len(data_bg_sig)):
    mc_bg_sig.append(tf1_bg_sig_pdf.GetRandom())
mc_bg_sig_fitter = ff.NLLFitter(bg_sig_model, np.asarray(mc_bg_sig))
mc_bg_sig_result = mc_bg_sig_fitter.fit([0.01,125.0, 2,0.0, 0.0, 0.0])

# plot some outputs

plt.figure()
plt.hist(data_bg,nbins,range=xlimits,alpha=0.2, histtype='stepfilled')
plt.plot(x,(len(data_bg)*binning)*bg_pdf(x,bg_result.x),linewidth=2)
plt.xlabel('M (GeV)')
plt.ylabel('Count')
plt.title ('Background-only distribution')

plt.figure()
plt.hist(data_sig,nbins,range=xlimits,alpha=0.2, histtype='stepfilled')
plt.plot(x,(len(data_sig)*binning)*sig_pdf(x,sig_result.x),linewidth=2)
plt.xlabel('M (GeV)')
plt.ylabel('Count')
plt.title ('Signal-only distribution')

plt.figure()
plt.hist(data_bg_sig,nbins,range=xlimits,alpha=0.2, histtype='stepfilled')
plt.plot(x,(len(data_bg_sig)*binning)*bg_sig_pdf(x,bg_sig_result.x),linewidth=2)
plt.xlabel('M (GeV)')
plt.ylabel('Count')
plt.title ('Signal+Background distribution')

plt.figure()
plt.hist(mc_bg_sig,nbins,range=xlimits,alpha=0.2, histtype='stepfilled')
plt.plot(x,(len(mc_bg_sig)*binning)*bg_sig_pdf(x,mc_bg_sig_result.x),linewidth=2)
plt.xlabel('M (GeV)')
plt.ylabel('Count')
plt.title ('Signal+Background MC Toy')

make_bb_plot(data_bg_sig, 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim',xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin',save_name='hgg_inject_hist')


#raw_input()
#tf1_bg_pdf = TF1("my_pdf",legendre_str,xlimits[0],xlimits[1])
#my_rands = []
#print len(data)

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

#make_bb_plot(hgg_bg_sm_range, 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_bg_hist')
#make_bb_plot(hgg_signal_selection, 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_signal_hist')
#edges = make_bb_plot(pd.concat([hgg_bg_sm_range,hgg_signal_selection],ignore_index=True), 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='P/bin',save_name='hgg_inject_hist')
#make_bb_plot(pd.concat([hgg_bg_sm_range,hgg_signal_selection],ignore_index=True), 0.02, bb_dir+'/plots/',title=r'pp$\to\gamma\gamma$ Sim', xlabel=r'$m_{\gamma\gamma}$ (GeV)', ylabel='A.U.',save_name='hgg_inject_hist',)
#plt.show()
plt.show()
