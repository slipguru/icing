#!/usr/bin/env python
"""Distance models available for icing.

The functions `score_dna`, `score_aa`,`char_dist_matrix` and `model_matrix`
are adapted from Change-O functions.
See changeo.DbCore for the original versions.
Reference: http://changeo.readthedocs.io/en/latest/

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import os
import numpy as np
import pandas as pd
import logging

from itertools import product


def score_dna(a, b, n_score=None, n_char='N', gap_score=None, gap='-.'):
    """Score a pair of IUPAC Ambiguous Nucleotide characters.

    Parameters
    ----------
    a, b : str
        Characters for which to calculate the match.
    n_score : None or float
        Score for all matches against a `N` character.
        If None, use IUPAC character identity.
    n_char : str, default 'N'
        N char.
    gap_score: None or float
        Score for all matches against a gap character;
        If None, use IUPAC character identity.
    gap : str, default '-.'
        Gap characters.

    Returns
    -------
    score : float
        Score for the two nucleotides.
    """
    # Define ambiguous character translations
    iupac_trans = {'AGWSKMBDHV': 'R', 'CTSWKMBDHV': 'Y', 'CGKMBDHV': 'S',
                   'ATKMBDHV': 'W', 'GTBDHV': 'K', 'ACBDHV': 'M',
                   'CGTDHV': 'B', 'AGTHV': 'D', 'ACTV': 'H', 'ACG': 'V',
                   'ABCDGHKMRSTVWY': 'N', '-.': '.'}
    # Create list of tuples of synonymous character pairs
    iupac_matches = [p for k, v in iupac_trans.iteritems()
                     for p in list(product(k, v))]

    # Check gap condition
    if gap_score is not None and (a in gap or b in gap):
        return gap_score

    # Check N-value condition
    if n_score is not None and (a == n_char or b == n_char):
        return n_score

    # Determine and return score for IUPAC match conditions
    # Symmetric and reflexive
    if a == b or (a, b) in iupac_matches or (b, a) in iupac_matches:
        return 1
    return 0


def score_aa(a, b, n_score=None, n_char='X', gap_score=None, gap='-.'):
    """Score a pair of IUPAC Extended Protein characters.

    Parameters
    ----------
    a, b : str
        Characters for which to calculate the match.
    n_score : None or float
        Score for all matches against a `N` character.
        If None, use IUPAC character identity.
    n_char : str, default 'X'
        N char.
    gap_score: None or float
        Score for all matches against a gap character;
        If None, use IUPAC character identity.
    gap : str, default '-.'
        Gap characters.

    Returns
    -------
    score : float
        Score for the two nucleotides.
    """
    # Define ambiguous character translations
    iupac_trans = {'RN': 'B', 'EQ': 'Z', 'LI': 'J',
                   'ABCDEFGHIJKLMNOPQRSTUVWYZ': 'X', '-.': '.'}
    # Create list of tuples of synonymous character pairs
    iupac_matches = [p for k, v in iupac_trans.iteritems()
                     for p in list(product(k, v))]

    # Check gap condition
    if gap_score is not None and (a in gap or b in gap):
        return gap_score

    # Check X-value condition
    if n_score is not None and (a == n_char or b == n_char):
        return n_score

    # Determine and return score for IUPAC match conditions
    # Symmetric and reflexive
    if a == b or (a, b) in iupac_matches or (b, a) in iupac_matches:
        return 1
    return 0


def char_dist_matrix(mat=None, n_score=0, gap_score=0, alphabet='dna'):
    """Generate a distance matrix between characters.

    Parameters
    ----------

    mat = input distance matrix to extend to full alphabet;
          if unspecified, creates Hamming distance matrix that incorporates
          IUPAC equivalencies
    n_score = score for all matches against an N character
    gap_score = score for all matches against a [-, .] character
    alphabet = the type of score dictionary to generate;
               one of [dna, aa] for DNA and amino acid characters

    Returns:
    a distance matrix (pandas DataFrame)
    """
    if alphabet == 'dna':
        iupac_chars = list('-.ACGTRYSWKMBDHVN')
        n = 'N'
        score_func = score_dna
    elif alphabet == 'aa':
        iupac_chars = list('-.*ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        n = 'X'
        score_func = score_aa
    else:
        logging.critical('Alphabet %s unrecognised.\n'.format(alphabet))

    # Default matrix to inf
    dist_mat = pd.DataFrame(float('inf'), index=iupac_chars,
                            columns=iupac_chars, dtype=float)
    # Set gap score
    for c in '-.':
        dist_mat.loc[c] = dist_mat.loc[:, c] = gap_score
    # Set n score
    dist_mat.loc[n] = dist_mat.loc[:, n] = n_score
    # Fill in provided distances from input matrix
    if mat is not None:
        for i, j in product(mat.index, mat.columns):
            dist_mat.loc[i, j] = mat.loc[i, j]
    # If no input matrix, create IUPAC-defined Hamming distance
    else:
        for i, j in product(dist_mat.index, dist_mat.columns):
            dist_mat.loc[i, j] = 1 - score_func(i, j, n_score=1-n_score,
                                                gap_score=1-gap_score)

    return dist_mat


def model_matrix(model, n_score=0, gap_score=0):
    """Get char dist matrix from model name.

    Parameters
    ----------
    model : ('ham', 'aa', 'hs1f', 'smith96' or 'm1n', 'hs5f')
        Model for character differences.
    n_score : float
        Score to assign to 'N' characters.
    gap_score : float
        Score to assign to GAP characters.

    Returns
    -------
    dist_mat : pandas.DataFrame
        Distance matrix between characters.
    """
    if model == 'aa':
        # Amino acid Hamming distance
        # n_score is overridden in case of aa
        aa_model = char_dist_matrix(n_score=1, gap_score=gap_score, alphabet='aa')
        return aa_model
    elif model == 'blosum50':
        model_path = os.path.dirname(os.path.realpath(__file__))
        blosum50_file = os.path.join(model_path, 'blosum50.csv')
        blosum50 = pd.read_csv(blosum50_file, header=0, index_col=0)  # in [-5,15]
        blosum50 += abs(np.min(blosum50.as_matrix()))  # now in [0,20]
        # it is a similarity score. Convert it to distance score.
        return np.max(blosum50.as_matrix()) - blosum50
    elif model == 'pam30':
        model_path = os.path.dirname(os.path.realpath(__file__))
        pam30_file = os.path.join(model_path, 'pam30.csv')
        pam30 = pd.read_csv(pam30_file, header=0, index_col=0)
        pam30 += abs(np.min(pam30.as_matrix()))
        # it is a similarity score. Convert it to distance score.
        return np.max(pam30.as_matrix()) - pam30
    elif model == 'ham':
        # DNA Hamming distance
        ham_model = char_dist_matrix(n_score=n_score, gap_score=gap_score, alphabet='dna')
        return ham_model
    elif model in ('m1n', 'smith96'):
        # Mouse 1-mer model
        smith96 = pd.DataFrame([[0.00, 2.86, 1.00, 2.14],
                                [2.86, 0.00, 2.14, 1.00],
                                [1.00, 2.14, 0.00, 2.86],
                                [2.14, 1.00, 2.86, 0.00]],
                               index=['A', 'C', 'G', 'T'],
                               columns=['A', 'C', 'G', 'T'], dtype=float)
        m1n_model = char_dist_matrix(smith96, n_score=n_score, gap_score=gap_score)
        return m1n_model
    elif model == 'hs1f':
        # Human 1-mer model
        hs1f = pd.DataFrame([[0.00, 2.08, 1.00, 1.75],
                             [2.08, 0.00, 1.75, 1.00],
                             [1.00, 1.75, 0.00, 2.08],
                             [1.75, 1.00, 2.08, 0.00]],
                            index=['A', 'C', 'G', 'T'],
                            columns=['A', 'C', 'G', 'T'], dtype=float)
        hs1f_model = char_dist_matrix(hs1f, n_score=n_score, gap_score=gap_score)
        return hs1f_model
    elif model == 'hs5f':
        # Human 5-mer DNA model
        model_path = os.path.dirname(os.path.realpath(__file__))
        hs5f_file = os.path.join(model_path, 'HS5F_Distance.tab')
        hs5f_model = pd.read_csv(hs5f_file, sep='\t', index_col=0)
        return hs5f_model
    else:
        raise ValueError('Unrecognized distance model: %s.\n', model)
