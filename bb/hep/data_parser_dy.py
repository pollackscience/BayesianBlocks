#! /usr/bin/env python

import cPickle as pkl
from root_pandas import read_root

zll = read_root('../../files/DY/Tree_LowPtSUSY_Tree_Delphes_ZLL_All_v2.root','MLL_Tree')
pkl.dump(zll, open( "../../files/DY/ZLL_v2.p", "wb" ), protocol = -1)

