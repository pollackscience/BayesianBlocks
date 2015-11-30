#! /usr/bin/env python

#from bayesianBlocks import *
import numpy as np
from hist_tools_modified import hist
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pkl
import os
from scipy.stats import powerlaw
from scipy.stats import triang
from scipy.stats import uniform



def generateToy():

  np.random.seed(12345)

  fig,ax = plt.subplots(4,sharex=True)
  #fig,ax = plt.subplots(2)

  powerlaw_arg = 2
  triang_arg=0.7
  n_samples = 500
  #generate simple line with slope 1, from 0 to 1
  frozen_powerlaw = powerlaw(powerlaw_arg) #powerlaw.pdf(x, a) = a * x**(a-1)
  #generate triangle with peak at 0.7
  frozen_triangle = triang(triang_arg) #up-sloping line from loc to (loc + c*scale) and then downsloping for (loc + c*scale) to (loc+scale).
  frozen_uniform = uniform(0.2,0.5)
  frozen_uniform2 = uniform(0.3,0.2)

  x = np.linspace(0,1)

  signal = np.random.normal(0.5, 0.1, n_samples/2)

  data_frame = pd.DataFrame({'powerlaw':powerlaw.rvs(powerlaw_arg,size=n_samples),
    'triangle':triang.rvs(triang_arg,size=n_samples),
    'uniform':np.concatenate((uniform.rvs(0.2,0.5,size=n_samples/2),uniform.rvs(0.3,0.2,size=n_samples/2))),
    'powerlaw_signal':np.concatenate((powerlaw.rvs(powerlaw_arg,size=n_samples/2),signal))})

  ax[0].plot(x, frozen_powerlaw.pdf(x), 'k-', lw=2, label='powerlaw pdf')
  hist(data_frame['powerlaw'],bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='100 bins',ax=ax[0])
  #hist(data_frame['powerlaw'],bins='blocks',fitness='poly_events',normed=True,histtype='stepfilled',alpha=0.2,label='b blocks',ax=ax[0])
  ax[0].legend(loc = 'best')

  ax[1].plot(x, frozen_triangle.pdf(x), 'k-', lw=2, label='triangle pdf')
  hist(data_frame['triangle'],bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='100 bins',ax=ax[1])
  hist(data_frame['triangle'],bins='blocks',fitness='poly_events',normed=True,histtype='stepfilled',alpha=0.2,label='b blocks',ax=ax[1])
  ax[1].legend(loc = 'best')

  #ax[0].plot(x, frozen_powerlaw.pdf(x), 'k-', lw=2, label='powerlaw pdf')
  hist(data_frame['powerlaw_signal'],bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='100 bins',ax=ax[2])
  #hist(data_frame['powerlaw_signal'],bins='blocks',normed=True,histtype='stepfilled',alpha=0.2,label='b blocks',ax=ax[2])
  ax[2].legend(loc = 'best')

  ax[3].plot(x, frozen_uniform.pdf(x)+frozen_uniform2.pdf(x), 'k-', lw=2, label='uniform pdf')
  hist(data_frame['uniform'],bins=100,normed=True,histtype='stepfilled',alpha=0.2,label='100 bins',ax=ax[3])
  #hist(data_frame['uniform'],bins='blocks',fitness = 'poly_events',p0=0.05,normed=True,histtype='stepfilled',alpha=0.2,label='b blocks',ax=ax[3])
  ax[3].legend(loc = 'best')

  plt.show()
  fig.savefig('plots/toy_plots.png')

  #x = np.random.normal(size=1000)
#  z_data_subset = z_data[0:20000]
#  plot_range = [50,400]
#  print 'max',max(z_data_subset),'min',min(z_data_subset)
#  plt.yscale('log', nonposy='clip')
#  plt.axes().set_ylim(0.0000001,0.17)
#  hist(z_data_subset,range=plot_range,bins=100,normed=1,histtype='stepfilled',
#      color=['lightgrey'], label=['100 bins'])
#  hist(z_data_subset,range=plot_range,bins='knuth',normed=1,histtype='step',linewidth=1.5,
#      color=['navy'], label=['knuth'])
#  hist(z_data_subset,range=plot_range,bins='blocks',normed=1,histtype='step',linewidth=2.0,
#      color=['crimson'], label=['b blocks'])
#  plt.legend()
#  #plt.yscale('log', nonposy='clip')
#  #plt.axes().set_ylim(0.0000001,0.17)
#  plt.xlabel(r'$m_{\ell\ell}$ (GeV)')
#  plt.ylabel('A.U.')
#  plt.title(r'Z$\to\mu\mu$ Data')
#  plt.savefig('z_data_hist_comp.png')
#  plt.show()
#

if __name__ == '__main__':
  generateToy()

