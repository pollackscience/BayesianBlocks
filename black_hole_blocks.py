#! /usr/bin/env python

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from hist_tools_modified import hist
import re
import cPickle as pkl

from ROOT import TFile
from ROOT import gDirectory

plt.close('all')
normed = True
log = True
try:
    ST_dict = pkl.load(open( "files/BHTree.p", "rb" ))
    print 'pickle loaded'
except:
    plt.close('all')
    infile = TFile('files/BHTree.root')
    intree = infile.Get('BH_Tree')
    entries = intree.GetEntriesFast()
    ST_dict = {}
    for i in range(2,11):
        ST_dict['ST_mul{}'.format(i)] = []

    for i in xrange(entries):
        if i%10000==0: print i
        ientry = intree.LoadTree(i)
        branch = intree.GetEntry(ientry)
        ST_dict['ST_mul2'].append(intree.ST_mul2)
        ST_dict['ST_mul3'].append(intree.ST_mul3)
        ST_dict['ST_mul4'].append(intree.ST_mul4)
        ST_dict['ST_mul5'].append(intree.ST_mul5)
        ST_dict['ST_mul6'].append(intree.ST_mul6)
        ST_dict['ST_mul7'].append(intree.ST_mul7)
        ST_dict['ST_mul8'].append(intree.ST_mul8)
        ST_dict['ST_mul9'].append(intree.ST_mul9)
        ST_dict['ST_mul10'].append(intree.ST_mul10)
    infile.Close()
    pkl.dump(ST_dict, open( "files/BHTree.p", "wb" ), protocol = -1)


for key in ST_dict.iterkeys():
    if key != 'ST_mul5': continue
    ST_dict[key]=[i for i in  ST_dict[key] if i>1900]
    print len(ST_dict[key])
    fig = plt.figure()
    ax = plt.gca()
    hist(ST_dict[key], bins=200, histtype='bar',alpha=0.2, label='standard histogram',normed=normed,log=log)
    print 'doing bb'
    p0 = int(re.sub("\D",'',key))
    p0=0.05
    hist(ST_dict[key], 'blocks', fitness = 'events', p0=p0,  ax = ax, histtype='step', label='Bayesian Blocks', linewidth=2,normed=normed,log=log)
    ax.legend()
    plt.title(key)
    plt.xlabel('ST (GeV)')
    if normed:
        plt.ylabel(r'N/$\Sigma$N$\Delta$x')
    else:
        plt.ylabel('N')
    plt.savefig('plots/'+key+'.pdf')
    plt.show()




