#! /usr/bin/env python

import cPickle as pkl
import pandas as pd
from root_pandas import read_root

df_mc = read_root('files/BH_Tree_QCD_HT-1000_inf_25ns_Double.root','BH_Tree',columns=['ST*','weightTree'])
df_data = read_root('files/BHTree.root','BH_Tree',columns=['ST*'])
pkl.dump(df_data, open( "files/BHTree_data.p", "wb" ), protocol = -1)
pkl.dump(df_mc, open( "files/BHTree_mc.p", "wb" ), protocol = -1)

