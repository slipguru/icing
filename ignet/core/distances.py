#!/usr/bin/env python
"""Utilities to compute distances between sequences.

The functions `get_nmers`, `single_distance` and `junction_distance` are
adapted from Change-O functions. See changeo.DbCore for the original version.
Reference: http://changeo.readthedocs.io/en/latest/
"""
from __future__ import division

import numpy as np

from itertools import izip  # combinations, izip, product
from Bio.pairwise2 import align

from ..align import align as igalign
from ..utils import extra

__author__ = 'Federico Tomasi'


def hamming(str1, str2):
    """Compute the hamming distances between two strings of equal length."""
    # return (np.array(list(str1)) != np.array(list(str2))).mean()
    return (np.fromstring(str1, np.int8) != np.fromstring(str2, np.int8)).mean()


def get_nmers(sequences, n):
    """Break sequences down into n-mers.

    Parameters
    ----------
    sequences : array-like
        String sequences.
    n : int
        Choose how to break down sequences. Usually is 1 or 5. Length of n-mers
        to return.

    Returns
    -------
    nmers : dictionary
        Dictionary built as: {sequence: [n-mers]}
    """
    sequences_n = ['N'*((n-1)/2) + seq + 'N'*((n-1)/2) for seq in sequences]
    nmers = {}
    for seq, seqn in izip(sequences, sequences_n):
        nmers[seq] = [seqn[i:i+n] for i in range(len(seqn)-n+1)]
    return nmers


def single_distance(seq1, seq2, n, dist_mat, norm, sym, mutations, tol=3,
                    c=35., length_constraint=True):
    """Calculate a distance between two input sequences.

    .. note:: Deprecated.
          `single_distance` will be removed in ignet 0.2. It is replaced by
          `string_distance`.

    :param seq1: first sequence
    :param seq2: second sequence
    :param n: length of n-mers to be used in calculating distance
    :param dist_mat: pandas DataFrame of mutation distances
    :param norm: normalization method
    :param sym: symmetry method
    :return: numpy matrix of pairwise distances between input sequences
    """
    import re
    if length_constraint and 0 < abs(len(seq1)-len(seq2)) <= tol:
        # different lengths, seqs alignment
        seq1, seq2 = map((lambda x: re.sub('[\.-]', 'N', str(x))),
                         align.globalxx(seq1, seq2)[0][:2])

    nmers = get_nmers([seq1, seq2], n)
    # Iterate over combinations of input sequences
    mutated = [i for i, (c1, c2) in enumerate(izip(seq1, seq2)) if c1 != c2]
    seqq1, seqq2 = ['']*len(mutated), ['']*len(mutated)
    nmer1, nmer2 = ['']*len(mutated), ['']*len(mutated)
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
    else:
        _dist = sum([sym_fun([float(dist_mat.at[c1,n2]),float(dist_mat.at[c2,n1])]) \
                 for c1,c2,n1,n2 in izip(seqq1,seqq2,nmer1,nmer2)]) / (norm_by)
        if mutations:
            try:
                alpha_mut = np.poly1d(np.load("polyfit_arguments_2.npy"))
                _dist *= (np.max(alpha_mut(np.max([mutations])), 0) + 0.2)
            except:
                _dist *= np.exp(-np.max(mutations) / c)
        return _dist


def junction_distance(seq1, seq2, n, dist_mat, norm, sym, tol=3, c=35.,
                      length_constraint=True):
    """Calculate a distance between two input sequences.

    .. note:: Deprecated.
          `junction_distance` will be removed in ignet 0.2. It is replaced by
          `string_distance`.

    Parameters
    ----------
    seq1, seq2 : str
        String sequences.
    n : int
        Choose how to break down sequences. Usually is 1 or 5.
    dist_mat : pandas.DataFrame
        Matrix which define the distance between the single characters.
    norm : ('len', 'mut', 'max', 'min', 'none')
        Normalisation method.
    sym : ('avg', 'min', 'sum')
        Choose how to symmetrise distances between seq1 and seq2 or seq2 and
        seq1.
    tol : int, optional, default: 3
        Tolerance in the length of the sequences. Default is 3 (3 nucleotides
        form an amminoacid. If seq1 and seq2 represent amminoacidic sequences,
        use tol = 1).
    c : float, optional, default: 35.0, deprecated
        Constant used with mutations. Now ignored. Will be removed.
    length_constraint : boolean, optional, default: True
        Insert the constraint on the difference between the lengths of seq1 and
        seq2. If False, `tol` is ignored.

    Returns
    -------
    distance : float
        A normalised distance between seq1 and seq2. Values are in [0,1].
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

    return sum([sym_fun([float(dist_mat.at[c1, n2]), float(dist_mat.at[c2, n1])])
                for c1, c2, n1, n2 in izip(seqq1, seqq2, nmer1, nmer2)]) / (norm_by)


def string_distance(seq1, seq2, dist_mat, norm_by=1, tol=3,
                    length_constraint=True):
    """Calculate a distance between two input sequences.

    Parameters
    ----------
    seq1, seq2 : str
        String sequences.
    dist_mat : pandas.DataFrame
        Matrix which define the distance between the single characters.
    norm_by : float, deprecated
        Normalising value for the distance.
    tol : int, optional, default: 3
        Tolerance in the length of the sequences. Default is 3 (3 nucleotides
        form an amminoacid. If seq1 and seq2 represent amminoacidic sequences,
        use tol = 1).
    length_constraint : boolean, optional, default: True
        Insert the constraint on the difference between the lengths of seq1 and
        seq2. If False, `tol` is ignored.

    Returns
    -------
    distance : float
        A normalised distance between seq1 and seq2. Values are in [0,1].
    """
    if length_constraint:
        l1, l2 = len(seq1), len(seq2)
        if abs(l1 - l2) > tol:
            return 1.  # min(l1, l2) / norm_by  # should be 1

        if 0 < abs(l1 - l2) <= tol:
            # different lengths, seqs alignment
            # seq1, seq2 = map(extra.junction_re, align.globalxx(seq1, seq2)[0][:2])
            seq1, seq2 = map(extra.junction_re, igalign.alignment(seq1, seq2))
    norm_by = len(seq1) * np.max(dist_mat.as_matrix())
    # return sum([np.mean((float(dist_mat.at[c1, c2]), float(dist_mat.at[c2, c1]))) for c1, c2 in izip(list(seq1), list(seq2)) if c1 != c2]) / norm_by
    return sum([np.mean((float(dist_mat.at[c1, c2]), float(dist_mat.at[c2, c1]))) for c1, c2 in izip(list(seq1), list(seq2))]) / norm_by
