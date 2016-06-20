#! /usr/bin/env python

import os
import cPickle as pkl
from root_pandas import read_root

current_dir = os.path.dirname(__file__)
bb_dir      = os.path.join(current_dir, '../..')
hgg_bg      = pkl.load(open(bb_dir+'/files/hgg_bg.p', "rb"))

df_data = read_root(bb_dir+'/files/BH/OutputFile_ForBrian.root','BH_Tree')
pkl.dump(df_data, open( bb_dir+"/files/BH/BH_paper_data.p", "wb" ), protocol = -1)

