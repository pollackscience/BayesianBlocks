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
ST_low = 2300

df_mc = pkl.load(open('files/BHTree_mc.p','rb'))
df_data = pkl.load(open('files/BHTree_data.p','rb'))

for ST in [10]:
    my_ST = df_mc[df_mc['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].values
    my_weights = df_mc[df_mc['ST_mul'+str(ST)]>ST_low]['weightTree'].values
    fig = plt.figure()
    ax = plt.gca()
    hist(my_ST, weights = my_weights, bins=200, histtype='bar',alpha=0.2, label='standard histogram',normed=normed,log=log)
    print 'doing bb'
    p0=0.01
    hist(my_ST, weights = my_weights, bins = 'blocks', fitness = 'events', p0=p0,  ax = ax, histtype='step', label='Bayesian Blocks', linewidth=2,normed=normed,log=log)
    ax.legend()
    plt.title('ST_mul'+str(ST))
    plt.xlabel('ST (GeV)')
    if normed:
        plt.ylabel(r'N/$\Sigma$N$\Delta$x')
    else:
        plt.ylabel('N')
    plt.savefig('plots/ST_mul'+str(ST)+'_MC.pdf')
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
