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

  fig,ax = plt.subplots()
  triang_arg=0.5
  #frozen_triangle = triang(c=triang_arg, loc=2) #up-sloping line from loc to (loc + c*scale) and then downsloping for (loc + c*scale) to (loc+scale).
  frozen_triangle = triang(c=0.5,loc=2) #up-sloping line from loc to (loc + c*scale) and then downsloping for (loc + c*scale) to (loc+scale).
  frozen_powerlaw = powerlaw(2) #powerlaw.pdf(x, a) = a * x**(a-1)

  x = np.linspace(0,1,20)
  x2 = np.linspace(0,1,20)
  nx = x
  nx2 = x2
  #nd = frozen_powerlaw.ppf(nx)
  #nd = np.array([0,0.3162,0.4472,0.5477,0.6324,0.7071,0.7746,0.8367,0.8944,0.9487])
  nd = np.array([0,0.140175,0.264911,0.378405,0.48324,0.581139,0.67332,0.760682,0.843909,0.923538])
  nd = np.array([0.0723805,0.204159,0.322876,0.431782,0.532971,0.627882,0.717556,0.802776,0.884144,0.962142])
  #pdf = frozen_powerlaw.pdf(x)
  #nd = frozen_triangle.ppf(nx)
  #print x
  #print nd
  #raw_input()
  #pdf = frozen_triangle.pdf(x)
  #print nd
  #print pdf
  #raw_input()
  #for i in range(len(nd)-1):
  #  print (nd[i+1]-nd[i])*(nd[i+1]+nd[i])
  #raw_input()

  #nd2 = frozen_triangle2.ppf(nx2)
  #pdf2 = frozen_triangle2.pdf(x2)

  #print nd,nd2
  #ndc = np.concatenate((nd,nd2),axis=0)
  #print 'ndc', ndc
  #nxc = np.concatenate((nx,nx2))
  #print pdf, pdf2
  #pdfc = np.concatenate((pdf,pdf2))
  #xc = np.concatenate((x,x2))

  #plt.plot(nd,len(nx)*[1],"x")
  #plt.plot(x,pdf)
  #hist(nd,'blocks',fitness='poly_events',p0=0.05,histtype='bar',alpha=0.2,label='b blocks',ax=ax,normed=True)

  #plt.plot(nd[0:11],len(nx[0:11])*[1],"x")
  #plt.plot(x[0:11],pdf[0:11])
  #hist(nd[0:11],'blocks',fitness='poly_events',p0=0.05,histtype='bar',alpha=0.2,label='b blocks',ax=ax,normed=True)
  #hist(ndc,bins=50,histtype='bar',alpha=0.2,label='b blocks',ax=ax,normed=True)

  #plt.plot(nd[11:],len(nx[11:])*[1],"x")
  #plt.plot(x[11:],pdf[11:])
  #hist(nd[11:],'blocks',fitness='poly_events',p0=0.05,histtype='bar',alpha=0.2,label='b blocks',ax=ax,normed=True)

  print nd
  plt.plot(nd,len(nd)*[1],"x")
  #plt.plot(x,pdf)
  hist(nd,'blocks',fitness='poly_events',p0=0.05,histtype='bar',alpha=0.2,label='b blocks',ax=ax)

  plt.show()
  fig.savefig('plots/toy_plots2.png')

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

