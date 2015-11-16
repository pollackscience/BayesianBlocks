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
samples = 500
seed = 1

df_mc = pkl.load(open('files/BHTree_mc.p','rb'))
df_data = pkl.load(open('files/BHTree_data.p','rb'))

#weights: array([ 0.27436519,  0.04019762,  0.01657276])
df_mc1 = df_mc[np.isclose(df_mc.weightTree,0.27436519)]
df_mc2 = df_mc[np.isclose(df_mc.weightTree, 0.0401976)]
df_mc3 = df_mc[np.isclose(df_mc.weightTree, 0.01657276)]

#ratio: 42:5:1

for ST in [5]:
    #my_ST = df_mc1[df_mc1['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].sample(samples*42,random_state=seed,replace=False).values
    my_ST = []
    my_ST = np.append(my_ST, df_mc2[df_mc2['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].sample(samples*5,random_state=seed).values)
    my_ST = np.append(my_ST, df_mc3[df_mc3['ST_mul'+str(ST)]>ST_low]['ST_mul'+str(ST)].sample(samples*1,random_state=seed).values)
    fig = plt.figure()
    ax = plt.gca()
    hist(my_ST, bins=200, histtype='bar',alpha=0.2, label='standard histogram',normed=normed,log=log)
    print 'doing bb'
    p0=0.01
    hist(my_ST, bins = 'blocks', fitness = 'events', p0=p0,  ax = ax, histtype='step', label='Bayesian Blocks', linewidth=2,normed=normed,log=log)
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
