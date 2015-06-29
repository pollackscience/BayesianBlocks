#! /usr/bin/env python

#from bayesianBlocks import *
import numpy as np
from astroML.plotting import hist
from astroML.
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pkl
import os



def generateToy():

  print 'loading values'
  if not os.path.isfile('values2.p'):
    z_data = np.loadtxt('values2.dat')
    pkl.dump( z_data, open( 'values2.p', "wb" ),pkl.HIGHEST_PROTOCOL )
  else:
    z_data = pkl.load(open('values2.p',"rb"))
  print 'loaded'

  #x = np.random.normal(size=1000)
  z_data_subset = z_data[0:20000]
  plot_range = [50,400]
  print 'max',max(z_data_subset),'min',min(z_data_subset)
  plt.yscale('log', nonposy='clip')
  plt.axes().set_ylim(0.0000001,0.17)
  hist(z_data_subset,range=plot_range,bins=100,normed=1,histtype='stepfilled',
      color=['lightgrey'], label=['100 bins'])
  hist(z_data_subset,range=plot_range,bins='knuth',normed=1,histtype='step',linewidth=1.5,
      color=['navy'], label=['knuth'])
  hist(z_data_subset,range=plot_range,bins='blocks',normed=1,histtype='step',linewidth=2.0,
      color=['crimson'], label=['b blocks'])
  plt.legend()
  #plt.yscale('log', nonposy='clip')
  #plt.axes().set_ylim(0.0000001,0.17)
  plt.xlabel(r'$m_{\ell\ell}$ (GeV)')
  plt.ylabel('A.U.')
  plt.title(r'Z$\to\mu\mu$ Data')
  plt.savefig('z_data_hist_comp.png')
  plt.show()


if __name__ == '__main__':
  generateToy()

