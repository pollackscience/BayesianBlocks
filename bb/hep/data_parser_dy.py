#! /usr/bin/env python

import cPickle as pkl
from root_pandas import read_root

zll = read_root('../../files/DY/ZLL.root','MLL_Tree', columns = 'Mll')
pkl.dump(zll, open( "../../files/DY/ZLL.p", "wb" ), protocol = -1)

