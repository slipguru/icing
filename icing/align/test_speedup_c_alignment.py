#!/usr/bin/env python
"""Test for speedup in using the alignment function in this folder.

This file is to visualise the speedup in using the alignment provided in
this folder w.r.t. Bio.pairwise2, also with the non-documented parameters
force_generic=True and one_alignment_only=True.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from Bio.pairwise2 import align

# from ignet.parallel import d_matrix_omp
from ignet.parallel_distance import dm_dense_inter_padding
from ignet.align import align as igalign


def _f(a, b):
    (align.globalxx(a, b, force_generic=True, one_alignment_only=True))
    return (max(len(a), len(b)))


def _g(a, b):
    (igalign.alignment(a, b))
    return (max(len(a), len(b)))

x = [50, 80, 150, 300, 500, 1000]
# x = [4]
y1 = []
y2 = []
for n in x:
    l1 = np.array(['aabbccddZZ'] * n)
    l2 = np.array(['bbFF'] * n)

    tic = time.time()
    dm_dense_inter_padding(l1, l2, _f)
    tac = time.time()
    # print(tac-tic)
    y1.append(tac-tic)

    tic = time.time()
    dm_dense_inter_padding(l1, l2, _g)
    tac = time.time()
    # print(tac-tic)
    y2.append(tac-tic)

plt.plot(x, y1, label="biopython alignment")
plt.plot(x, y2, label="ignet alignment")
plt.legend()
plt.show()
