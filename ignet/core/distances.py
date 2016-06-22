#!/usr/bin/env python
"""Utilities to compute distances between sequences."""
from __future__ import division

import numpy as np
from itertools import izip  # combinations, izip, product
from Bio.pairwise2 import align

from ..utils import extra


def hamming(str1, str2):
    """Compute the hamming distances between two strings of equal length."""
    # return (np.array(list(str1)) != np.array(list(str2))).mean()
    return (np.fromstring(str1, np.int8) != np.fromstring(str2, np.int8)).mean()


def get_nmers(sequences, n):
    """Break input sequences down into n-mers.

    :param sequences: list of sequences to be broken into n-mers
    :param n: length of n-mers to return
    :return: dictionary of {sequence: [n-mers]}
    """
    sequences_n = ['N' * ((n-1)/2) + seq + 'N' * ((n-1)/2) for seq in sequences]
    nmers = {}
    for seq, seqn in izip(sequences, sequences_n):
        nmers[seq] = [seqn[i:i+n] for i in range(len(seqn)-n+1)]
    return nmers


def junction_distance(seq1, seq2, n, dist_mat, norm, sym, mutations=None, tol=3,
                      c=35., length_constraint=True):
    """Calculate a distance between two input sequences.

    :param seq1: first sequence
    :param seq2: second sequence
    :param n: length of n-mers to be used in calculating distance
    :param dist_mat: pandas DataFrame of mutation distances
    :param norm: normalization method
    :param sym: symmetry method
    :return: numpy matrix of pairwise distances between input sequences
    """
    if length_constraint and 0 < abs(len(seq1)-len(seq2)) <= tol:
        # different lengths, seqs alignment
        seq1, seq2 = map(extra.junction_re, align.globalxx(seq1, seq2)[0][:2])

    nmers = get_nmers([seq1, seq2], n)
    mutated = np.array([i for i, (c1, c2) in enumerate(izip(seq1, seq2)) if c1 != c2])
    mut_len = mutated.shape[0]
    seqq1 = np.empty(mut_len, dtype=object)
    seqq2 = np.empty(mut_len, dtype=object)
    nmer1 = np.empty(mut_len, dtype=object)
    nmer2 = np.empty(mut_len, dtype=object)
    for i, m in enumerate(mutated):
        seqq1[i] = seq1[m]
        seqq2[i] = seq2[m]
        nmer1[i] = nmers[seq1][m]
        nmer2[i] = nmers[seq2][m]

    # Determine normalizing factor
    if norm == 'len':
        norm_by = len(seq1)
    elif norm == 'mut':
        norm_by = len(mutated)
    elif norm == 'max':
        norm_by = max(len(seq1), len(seq2))
    elif norm == 'min':
        norm_by = min(len(seq1), len(seq2))
    else:
        norm_by = 1

    # Determine symmetry function
    if sym == 'avg':
        sym_fun = np.mean
    elif sym == 'min':
        sym_fun = min
    else:
        sym_fun = sum

    if length_constraint and abs(len(seq1)-len(seq2)) > tol:
        return min(len(seq1), len(seq2)) / norm_by

    return sum([sym_fun([float(dist_mat.at[c1,n2]),float(dist_mat.at[c2,n1])]) \
                for c1,c2,n1,n2 in izip(seqq1,seqq2,nmer1,nmer2)]) / (norm_by)


def string_distance(seq1, seq2, dist_mat, norm_by, tol=3, length_constraint=True):
    """Calculate a distance between two input sequences.

    :param seq1: first sequence
    :param seq2: second sequence
    :param dist_mat: pandas DataFrame of mutation distances
    :param norm: normalization method
    :param sym: symmetry method
    :return: numpy matrix of pairwise distances between input sequences
    """
    if length_constraint:
        l1, l2 = len(seq1), len(seq2)
        if abs(l1 - l2) > tol:
            return 1.  # min(l1, l2) / norm_by  # should be 1

        if 0 < abs(l1 - l2) <= tol:
            # different lengths, seqs alignment
            seq1, seq2 = map(extra.junction_re, align.globalxx(seq1, seq2)[0][:2])
            norm_by = len(seq1)

    return sum([np.mean((float(dist_mat.at[c1,c2]), float(dist_mat.at[c2,c1]))) for c1, c2 in izip(list(seq1),list(seq2)) if c1 != c2]) / norm_by
