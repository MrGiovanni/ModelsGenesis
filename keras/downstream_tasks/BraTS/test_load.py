#!/usr/bin/env python
"""
File: test_load
Date: 5/4/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import io
import BraTS
import timeit
import cProfile, pstats

BraTS.set_root("/Users/jonpdeaton/Datasets/BraTS")
brats = BraTS.DataSet(year=2017)


def load():
    subset = brats.train.subset(brats.train.ids[:10])
    x = subset.mris


def load_n(n=10):
    for i in range(n):
        p = brats.train.patient(brats.train.ids[i])


s = timeit.timeit(load)
print("Time: %s sec" % s)



