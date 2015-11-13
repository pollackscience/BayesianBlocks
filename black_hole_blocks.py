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
    ST_dict_data = pkl.load(open( "files/BHTree.p", "rb" ))
    print 'pickle loaded'
except:
    plt.close('all')
    infile = TFile('files/BHTree.root')
    intree = infile.Get('BH_Tree')
    entries = intree.GetEntriesFast()
    ST_dict_data = {}
    for i in range(2,11):
        ST_dict_data['ST_mul{}'.format(i)] = []

    for i in xrange(entries):
        if i%10000==0: print i
        ientry = intree.LoadTree(i)
        branch = intree.GetEntry(ientry)
        ST_dict_data['ST_mul2'].append(intree.ST_mul2)
        ST_dict_data['ST_mul3'].append(intree.ST_mul3)
        ST_dict_data['ST_mul4'].append(intree.ST_mul4)
        ST_dict_data['ST_mul5'].append(intree.ST_mul5)
        ST_dict_data['ST_mul6'].append(intree.ST_mul6)
        ST_dict_data['ST_mul7'].append(intree.ST_mul7)
        ST_dict_data['ST_mul8'].append(intree.ST_mul8)
        ST_dict_data['ST_mul9'].append(intree.ST_mul9)
        ST_dict_data['ST_mul10'].append(intree.ST_mul10)
    infile.Close()
    pkl.dump(ST_dict_data, open( "files/BHTree.p", "wb" ), protocol = -1)

try:
    ST_dict_MC = pkl.load(open( "files/BHTree_QCD2000.p", "rb" ))
    print 'pickle loaded'
except:
    plt.close('all')
    infile = TFile('files/BH_Tree_QCD_HT-2000_inf_25ns.root')
    intree = infile.Get('BH_Tree')
    entries = intree.GetEntriesFast()
    ST_dict_MC = {}
    for i in range(2,11):
        ST_dict_MC['ST_mul{}'.format(i)] = []

    for i in xrange(entries):
        if i%10000==0: print i
        ientry = intree.LoadTree(i)
        branch = intree.GetEntry(ientry)
        ST_dict_MC['ST_mul2'].append(intree.ST_mul2)
        ST_dict_MC['ST_mul3'].append(intree.ST_mul3)
        ST_dict_MC['ST_mul4'].append(intree.ST_mul4)
        ST_dict_MC['ST_mul5'].append(intree.ST_mul5)
        ST_dict_MC['ST_mul6'].append(intree.ST_mul6)
        ST_dict_MC['ST_mul7'].append(intree.ST_mul7)
        ST_dict_MC['ST_mul8'].append(intree.ST_mul8)
        ST_dict_MC['ST_mul9'].append(intree.ST_mul9)
        ST_dict_MC['ST_mul10'].append(intree.ST_mul10)
    infile.Close()
    pkl.dump(ST_dict_MC, open( "files/BHTree_QCD2000.p", "wb" ), protocol = -1)


for key in ST_dict_data.iterkeys():
    #if key != 'ST_mul10': continue
    print len(ST_dict_data[key])
    ST_dict_data[key]=[i for i in  ST_dict_data[key] if i>1900]
    print key, len(ST_dict_data[key])
    a = ST_dict_data[key]
    fig = plt.figure()
    ax = plt.gca()
    hist(ST_dict_data[key], bins=200, histtype='bar',alpha=0.2, label='standard histogram',normed=normed,log=log)
    print 'doing bb'
    p0=0.01
    hist(ST_dict_data[key], 'blocks', fitness = 'events', p0=p0,  ax = ax, histtype='step', label='Bayesian Blocks', linewidth=2,normed=normed,log=log)
    ax.legend()
    plt.title(key)
    plt.xlabel('ST (GeV)')
    if normed:
        plt.ylabel(r'N/$\Sigma$N$\Delta$x')
    else:
        plt.ylabel('N')
    plt.savefig('plots/'+key+'.pdf')
    plt.show()

'''
for key in ST_dict_MC.iterkeys():
    #if key != 'ST_mul2': continue
    ST_dict_MC[key]=[i for i in  ST_dict_MC[key] if i>1900][:15000]
    print len(ST_dict_MC[key])
    fig = plt.figure()
    ax = plt.gca()
    hist(ST_dict_MC[key], bins=200, histtype='bar',alpha=0.2, label='standard histogram',normed=normed,log=log)
    print 'doing bb'
    p0 = int(re.sub("\D",'',key))
    p0=0.05
    hist(ST_dict_MC[key], 'blocks', fitness = 'events', p0=p0,  ax = ax, histtype='step', label='Bayesian Blocks', linewidth=2,normed=normed,log=log)
    ax.legend()
    plt.title(''.join([key,'QCD HT2000']))
    plt.xlabel('ST (GeV)')
    if normed:
        plt.ylabel(r'N/$\Sigma$N$\Delta$x')
    else:
        plt.ylabel('N')
    plt.savefig('plots/'+key+'_MC_.pdf')
    plt.show()
'''
