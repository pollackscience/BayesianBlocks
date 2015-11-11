#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from hist_tools_modified import hist
from scipy.stats import uniform

plt.close('all')
fig, ax = plt.subplots()
plt.show()
plt.get_current_fig_manager().window.wm_geometry("-2800-600")
ax.set_xlim(0,0.55)
ax.set_ylim(0,5.2)
test_data=[ 0.03815414 , 0.09320462 , 0.11173259 , 0.3899594 ,  0.48899476]
print test_data

my_hist = None
frame=0
for i in range(len(test_data)):
  frame+=1

  ax.plot(test_data[:i+1],len(test_data[:i+1])*[1],"o",zorder=40,color='k')
  if my_hist != None and len(my_hist)>0:
    for patch in my_hist:
      patch.set_alpha(0.15)
  plt.show()
  plt.savefig('plots/frame{}.png'.format(frame))
  #hist(test_data[:i+1],'blocks',fitness='events',p0=0.37,histtype='bar',alpha=0.2,label='b blocks',ax=ax,zorder=len(test_data)-i)
  #my_hists.append(hist(test_data[:i+1],'blocks',fitness='events',p0=0.37,histtype='bar',label='b blocks',ax=ax,zorder=10-i)[-1])
  my_hist = hist(test_data[:i+1],'blocks',fitness='events',p0=0.37,histtype='bar',label='b blocks',ax=ax,zorder=20-i)[-1]
  #print my_hist
  #plt.draw()
  plt.show()
  frame+=1
  plt.savefig('plots/frame{}.png'.format(frame))
  #raw_input()
  #for patch in hist:
  #  patch.set_alpha(0.2)
