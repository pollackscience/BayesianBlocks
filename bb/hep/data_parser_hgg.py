#! /usr/bin/env python

import cPickle as pkl
import pandas as pd
from root_pandas import read_root

df_signal = read_root('../../files/HiggsToGG/Tree_HToGG.root','HGG_Tree')
df_bg = read_root('../../files/HiggsToGG/Tree_PPToGG.root','HGG_Tree')
pkl.dump(df_signal, open( "../../files/hgg_signal.p", "wb" ), protocol = -1)
pkl.dump(df_bg, open( "../../files/hgg_bg.p", "wb" ), protocol = -1)

