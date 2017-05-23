#!/usr/bin/env python
"""Utilities to compute distances between sequences.

The functions `get_nmers` and `junction_distance` are
adapted from Change-O functions. See changeo.DbCore for the original version.
Reference: http://changeo.readthedocs.io/en/latest/

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import division

import numpy as np

from itertools import izip  # combinations, izip, product
from Bio.pairwise2 import align
from sklearn.base import BaseEstimator

# try:
#     from icing.align import align as igalign
# except ImportError:
#     raise ImportError("Module align.so not found. "
#                       "Did you compile icing with "
#                       "'python setup.py build_ext --inplace install'?")

from icing.kernel import stringkernel
# from string_kernel import stringkernel
from icing.models.model import model_matrix
from icing.utils import extra


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
    sequences_n = ['N'*int((n-1)/2) + seq + 'N'*int((n-1)/2) for seq in sequences]
    nmers = {}
    for seq, seqn in izip(sequences, sequences_n):
        nmers[seq] = [seqn[i:i+n] for i in range(len(seqn)-n+1)]
    return nmers


def junction_distance(seq1, seq2, n, dist_mat, norm, sym, tol=3, c=35.,
                      length_constraint=True):
    """Calculate a distance between two input sequences.

    .. note:: Deprecated.
          `junction_distance` will be removed in icing 0.2. It is replaced by
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


def string_distance(seq1, seq2, len_seq1, len_seq2, dist_mat, dist_mat_max,
                    tol=3, length_constraint=True):
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
        if abs(len_seq1 - len_seq2) > tol:
            return 1.  # min(len_seq1, len_seq2) / norm_by  # should be 1

        if 0 < abs(len_seq1 - len_seq2) <= tol:
            # different lengths, seqs alignment
            seq1, seq2 = map(extra.junction_re, align.globalms(
                seq1, seq2, 5, -4, -3, -.1)[0][:2])
            len_seq1 = len(seq1)
            # print 'befor align:\n', seq1, '\n', seq2, '\n--------------'
            # seq1, seq2 = map(extra.junction_re, igalign.alignment(seq1, seq2))
            # print 'after align:\n', seq1, '\n', seq2, '\n--------------'
    norm_by = len_seq1 * dist_mat_max
    return sum([np.mean((
        float(dist_mat.at[c1, c2]),
        float(dist_mat.at[c2, c1]))) for c1, c2 in izip(
            list(seq1), list(seq2))]) / norm_by


class Distance(BaseEstimator):
    _estimator_type = "distance"


class StringKernelDistance(Distance):
    """Utility class for string kernel for computing distances."""

    def __init__(self, min_kn=1, max_kn=2, lamda=.5,
                 check_min_length=0, hard_matching=0):
        self.min_kn = min_kn
        self.max_kn = max_kn
        self.lamda = lamda
        self.check_min_length = check_min_length
        self.hard_matching = hard_matching

    def pairwise(self, x1, x2):
        self.max_kn = min(min(len(x1), len(x2)), self.max_kn)
        return 1 - stringkernel(
            [x1, x2], verbose=False, normalize=1, return_float=1,
            min_kn=self.min_kn, max_kn=self.max_kn, lamda=self.lamda,
            check_min_length=self.check_min_length,
            hard_matching=self.hard_matching)


class StringDistance(Distance):
    """Utility class for string distance."""

    def __init__(self, model='ham', dist_mat=None, tol=3):
        self.model = model
        self.dist_mat = dist_mat
        self.tol = tol

        if self.dist_mat is None:
            self.dist_mat = model_matrix(model)
        self.dist_mat_max = np.max(np.max(self.dist_mat))

    def pairwise(self, x1, x2):
        return string_distance(
            x1, x2, len(x1), len(x2), dist_mat=self.dist_mat,
            dist_mat_max=self.dist_mat_max, tol=self.tol)


class IgDistance(Distance):
    """Container for computing distance between IgRecord string representation."""

    # def __init__(self, method='jaccard', model='ham', dist_mat=None, dist_mat_max=1,
    #     tol=3, rm_duplicates=False,
    #     v_weight=1., j_weight=1., vj_weight=.5, sk_weight=.5,
    #     correction_function=(lambda _: 1), correct=True,
    #     sim_score_params=None, ssk_params=None):
    #     pass
    def __init__(
            self, junction_dist, tol=3, rm_duplicates=False,
            correct=True, correct_by=None):
        """Calculate a similarity between two input immunoglobulins.

        Parameters
        ----------
        tol : int, optional, default: 3
            Tolerance in the length of the sequences. Default is 3 (3 nucleotides
            form an amminoacid. If seq1 and seq2 represent amminoacidic sequences,
            use tol = 1).

        Returns
        -------
        similarity : float
            A normalised similarity score between ig1 and ig2. Values are in [0,1].
            0: ig1 is completely different from ig2.
            1: ig1 and ig2 are the same.

        """
        self.junction_dist = junction_dist
        self.rm_duplicates = rm_duplicates
        self.correct = correct
        self.correct_by = correct_by
        self.tol = tol

    def pairwise(self, x1, x2):
        """Compute pairwise similarity.

        Parameters
        ----------
        x1, x2 : array-like
            String representation of two Igs. See IgRecords.features()
        """
        if self.rm_duplicates and x1[2] == x2[2]:
            return 1

        Vgenes_x, Jgenes_x = map(lambda _: set(_.split('|')), (x1[0], x1[1]))
        Vgenes_y, Jgenes_y = map(lambda _: set(_.split('|')), (x2[0], x2[1]))

        if abs(float(x1[3]) - float(x2[3])) > self.tol or len(
                Vgenes_x & Vgenes_y) < 1:
            return 1

        distance = self.junction_dist.pairwise(x1[2], x2[2])

        if distance > 0 and self.correct:
            correction = self.correct_by(np.mean((float(x1[4]), float(x2[4]))))
            distance *= np.clip(correction, 0, 1)
        return max(distance, 0)


def correction_function(x):
    return 1 - x / 100.


def distance_dataframe(s, x1, x2, rm_duplicates=False, tol=3,
                       junction_dist=None, correct=False,
                       correct_by=correction_function,
                       model='nt'):
    """Compute pairwise similarity.

    Parameters
    ----------
    x1, x2 : array-like
        String representation of two Igs. See IgRecords.features()
    """
    # let's use X as lookup table instead of data
    try:
        x1, x2 = s.iloc[int(x1[0])], s.iloc[int(x2[0])]
    except TypeError as e:
        print x1, x2
        raise e

    model = 'aa_' if model == 'aa' else ''
    x1_junc, x2_junc = x1[model + 'junc'], x2[model + 'junc']
    Vgenes_x, Jgenes_x = x1.v_gene_set, x1.j_gene_set
    Vgenes_y, Jgenes_y = x2.v_gene_set, x2.j_gene_set

    if x1_junc == x2_junc:
        if (Vgenes_x & Vgenes_y):
            if rm_duplicates:
                return 1
            return 0
        else:
            return 1

    if abs(x1[model + 'junction_length'] - x2[
            model + 'junction_length']) > tol or \
            not (Vgenes_x & Vgenes_y):
        return 1

    distance = junction_dist.pairwise(x1_junc, x2_junc)
    if 1 > distance > 0 and correct:
        correction = correct_by(np.max((x1.mut, x2.mut)))
        distance *= np.clip(correction, 0, 1)
    return max(distance, 0)


def ig_distance(s, x1, x2, rm_duplicates=False, tol=3, junction_dist=None,
                correct=False):
    """Compute pairwise similarity.

    Parameters
    ----------
    x1, x2 : array-like
        String representation of two Igs. See IgRecords.features()
    """
    # let's use X as lookup table instead of data
    try:
        x1, x2 = s[int(x1[0])], s[int(x2[0])]
    except TypeError as e:
        print x1, x2
        raise e

    if rm_duplicates and x1[2] == x2[2]:
        return 1

    Vgenes_x, Jgenes_x = map(lambda _: set(_.split('|')), (x1[0], x1[1]))
    Vgenes_y, Jgenes_y = map(lambda _: set(_.split('|')), (x2[0], x2[1]))

    if abs(float(x1[3]) - float(x2[3])) > tol or not (Vgenes_x & Vgenes_y):
        return 1

    distance = junction_dist.pairwise(x1[2], x2[2])

    if 1 > distance > 0 and correct:
        correction = correction_function(np.mean((float(x1[4]), float(x2[4]))))
        distance *= np.clip(correction, 0, 1)
    return max(distance, 0)


def is_distance(estimator):
    """Return True if the given estimator encode a distance."""
    return getattr(estimator, "_estimator_type", None) == "distance"
