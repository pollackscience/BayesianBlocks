#! /usr/bin/env python

from __future__ import division
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from astroML.plotting import hist
import scipy.stats as st

import ROOT
from ROOT import TFile
from ROOT import gDirectory
from ROOT import TF1
from ROOT import TRandom3

plt.close('all')
normed = True
log = True

ROOT.gRandom.SetSeed(8675309)
my_pdf = TF1("my_pdf","2.3949e6*3.74/(0.67472+x)**10.1809",1900,5500)
my_rands = []
for i in xrange(14653):
    my_rands.append(my_pdf.GetRandom())
    if i == 0: print my_rands[-1]

fig = plt.figure()
ax = plt.gca()
hist(my_rands, bins=200, histtype='bar',alpha=0.2, label='standard histogram', log=log, normed=normed)
hist(my_rands, bins='blocks', histtype='step',linewidth=2, label='bb',log=log, normed=normed)
plt.legend()
plt.title('MC gen from ST_mul5 fit function')
plt.xlabel('ST (GeV)')
if normed:
    plt.ylabel(r'N/$\Sigma$N$\Delta$x')
else:
    plt.ylabel('N')
plt.show()
plt.savefig('plots/ST_mul5_fit.pdf')


