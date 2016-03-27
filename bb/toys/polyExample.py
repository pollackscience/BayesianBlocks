#! /usr/bin/env python

#from bayesianBlocks import *
import numpy as np
from bb.tools.hist_tools_modified import hist
from bb.tools.bayesian_blocks_modified import bayesian_blocks
import bb.tools.bb_poly as bbp
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pkl
import os
from scipy.stats import powerlaw
from scipy.stats import triang
from scipy.stats import uniform
import ROOT
from ROOT import TF1



def generateToy():
    plt.close('all')

    def poly1(x):
        return 2*x/100

    nentries = 100
    p0=0.01
    x = np.arange(0.0, 10, 0.1)
    np.random.seed(12345)
    ROOT.gRandom.SetSeed(8675309)
    poly1_gen = TF1("poly1","2*x",0,10)
    my_rands = []
    for i in xrange(nentries):
        my_rands.append(poly1_gen.GetRandom())

    fig = plt.figure()
    hist(my_rands,bins=10,histtype='stepfilled',alpha=0.2,label='10 bins',normed=True)
    bb_edges = bayesian_blocks(my_rands,p0=p0)
    hist(my_rands,bins=bb_edges,histtype='stepfilled',alpha=0.2,label='10 bins',normed=True)
    plt.plot(x,poly1(x),'k')
    plt.show()

if __name__ == '__main__':
    generateToy()

