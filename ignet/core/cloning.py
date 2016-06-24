#!/usr/bin/env python
"""Assign Ig sequences into clones."""
from __future__ import print_function

import os
import sys
import imp
import numpy as np
import scipy
import multiprocessing as mp
import joblib as jl
import logging
import cPickle as pkl
import gzip

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.utils.sparsetools import connected_components

from .distances import string_distance  # , junction_distance
from .similarity_scores import similarity_score_tripartite as mwi
# from .. import parallel_distance
from ..externals import DbCore
from ..models.model import model_matrix
from ..plotting import silhouette
from ..utils import io
from ..utils import extra

__author__ = 'Federico Tomasi'


def alpha_mut(ig1, ig2, fn='../models/negexp_pars.npy'):
    """Coefficient to balance distance according to mutation levels."""
    def _neg_exp(x, a, c, d):
        return a * np.exp(-c * x) + d

    try:
        popt = np.load(fn)
        return _neg_exp(np.max((ig1.mut, ig2.mut)), *popt)
    except:
        return np.exp(-np.max((ig1.mut, ig2.mut)) / 35.)


def dist_function(ig1, ig2, method='jaccard', model='ham',
                  dist_mat=None, tol=3):
    """Calculate a distance between two input immunoglobulins.

    Parameters
    ----------
    ig1, ig2 : externals.DbCore.IgRecord
        Instances of two immunoglobulins.
    method : ('jaccard', 'simpson'), optional
        Graph-based index.
    model : ('ham', 'hs1f'), optional
        Model for the distance between strings.
    dist_mat : pandas.DataFrame, optional
        Matrix which define the distance between the single characters.
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
    # nmer = 5 if model == 'hs5f' else 1
    if ig1.junction_length - ig2.junction_length > tol:
        return 0.

    if not dist_mat:
        dist_mat = model_matrix(model)

    V_genes_ig1 = ig1.getVGene('set')
    V_genes_ig2 = ig2.getVGene('set')
    J_genes_ig1 = ig1.getJGene('set')
    J_genes_ig2 = ig2.getJGene('set')

    ss = mwi(V_genes_ig1, V_genes_ig2, J_genes_ig1, J_genes_ig2, method=method)
    if ss > 0.:
        # print V_genes_ig1, V_genes_ig2, J_genes_ig1, J_genes_ig2
        junc1 = extra.junction_re(ig1.junction)
        junc2 = extra.junction_re(ig2.junction)
        norm_by = min(len(junc1), len(junc2))
        if model == 'hs1f':
            norm_by *= 2.08
        dist = string_distance(junc1, junc2, dist_mat, norm_by, tol=tol)
        ss *= (1 - dist)
    if ss > 0:
        ss = 1 - ((1 - ss) * alpha_mut(ig1, ig2))
    return ss
# d_func = lambda x, y: dist_function(x, y, 'jaccard', 'ham')
# d_wrapper = lambda x, y: (d_func(x[1], y[1]), (x[0], y[0]))


def d_func(x, y):
    return dist_function(x, y, 'jaccard', 'ham')


def d_wrapper(x, y):
    return (d_func(x[1], y[1]), (x[0], y[0]))


def compute_similarity_matrix(db_iter, sparse_mode=True):
    """Compute the similarity matrix from a database iterator.

    Parameters
    ----------
    db_iter : generator
        Records loaded by the script `ig_run.py`.
    sparse_mode : bool, optional, default `True`
        Return a sparse similarity matrix.

    Returns
    -------
    similarity_matrix : scipy.sparse array or numpy.ndarray
        Similarity matrix between records in db_iter.

    Notes
    -----
    This function uses sparse matrices and an inverse index to efficiently
    compute the similarity matrix between a potentially long list of records.
    In ancient times, this function was simply a call to
     `parallel_distance.distance_matrix_parallel(db_iter, d_func, sparse_mode)`
    However, this method is really inefficient, so the matrix is computed
    between chosen couples of values.

    The reverse index with one core, instead, looks like
        r_index = dict()
        for i, ig in enumerate(igs):
            for v in ig.getVGene('set'): r_index.setdefault(v,[]).append((i,ig))
            for j in ig.getJGene('set'): r_index.setdefault(j,[]).append((i,ig))
    """
    def _get_v_j(d, i_igs):
        for i, ig in i_igs:
            for v in ig.getVGene('set'):
                # d.setdefault(v,[]).append((i,ig))
                d[v] = d.get(v, []) + [(i, )]
            for j in ig.getJGene('set'):
                # d.setdefault(j,[]).append((i,ig))
                d[j] = d.get(j, []) + [(i, )]

    def _get_v_j_padding(d, idx, nprocs, igs_arr, n):
        for i in range(idx, n, nprocs):
            ig = igs_arr[i]
            for v in ig.getVGene('set'):
                # d.setdefault(v,[]).append((i,ig))
                d[v] = d.get(v, []) + [(i, )]
            for j in ig.getJGene('set'):
                # d.setdefault(j,[]).append((i,ig))
                d[j] = d.get(j, []) + [(i, )]

    igs = list(db_iter)
    # igs_arr = np.array(igs)
    igs_arr = (igs)
    n = len(igs)
    logging.info("{} Igs read.".format(n))
    manager = mp.Manager()
    r_index = manager.dict()
    nprocs = min(n, mp.cpu_count())
    ps = []
    # for idx in range(nprocs):
        # p = mp.Process(target=_get_v_j_padding, args=(r_index, idx, nprocs, igs_arr, n))
    for i_igs in extra.split_list(list(enumerate(igs)), nprocs):
        p = mp.Process(target=_get_v_j, args=(r_index, i_igs))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

    print(r_index)
    indicator_matrix = scipy.sparse.lil_matrix((n, n),
                                               dtype=np.dtype('bool'))
    s1 = set()
    r_index = dict(r_index)
    for k, v in r_index.iteritems():
        length = len(v)
        if length > 1:
            for k in (range(int(length*(length-1)/2))):
                j = k%(length-1)+1; i = int(k/(length-1));
                if i >= j:
                    i = length-i-1; j=length-j
                s1.add((v[i][0], v[j][0]))
                s1.add((v[j][0], v[i][0]))
                indicator_matrix[v[i][0], v[j][0]] = True

    r_index = dict()
    for i, ig in enumerate(igs):
        for v in ig.getVGene('set'): r_index.setdefault(v,[]).append((i,))
        for j in ig.getJGene('set'): r_index.setdefault(j,[]).append((i,))

    s2 = set()
    r_index = dict(r_index)
    for k, v in r_index.iteritems():
        length = len(v)
        if length > 1:
            for k in (range(int(length*(length-1)/2))):
                j = k%(length-1)+1; i = int(k/(length-1));
                if i >= j:
                    i = length-i-1; j=length-j
                s2.add((v[i][0], v[j][0]))
                s2.add((v[j][0], v[i][0]))
                indicator_matrix[v[i][0], v[j][0]] = True

            # for i in range(length):
            #     for j in range(i+1, length):
    print(s1.difference(s2))
    print(s2.difference(s1))

    rows, cols, _ = map(list, scipy.sparse.find(indicator_matrix))
    data = jl.Parallel(n_jobs=-1)(jl.delayed(d_func)
                                  (igs[i], igs[j]) for i, j in zip(rows, cols))
    data = np.array(data)
    idx = data > 0
    data = data[idx]
    rows = np.array(rows)[idx]
    cols = np.array(cols)[idx]

    # inefficient:
    # S = set()
    # for k,v in r_index.iteritems():
    #     length = len(v)
    #     if length > 1:
    #         S.update([(v[i], v[j]) for i in range(length) for j in range(i+1, length)])

    # d_func = lambda x, y: dist_function(x, y, 'jaccard', 'ham')
    # d_wrapper = lambda x, y: (d_func(x[1], y[1]), (x[0], y[0]))

    # data, pos = zip(*jl.Parallel(n_jobs=mp.cpu_count())
    #            (jl.delayed(d_wrapper)(i,j) for (i,j) in S))
    # rows, cols = map(list, zip(*pos))
    # n = max(max(rows), max(cols))+1

    # sim = 1 - np.array(data)
    sparse_mat = scipy.sparse.csr_matrix((data, (rows, cols)),
                                         shape=(n, n))
    similarity_matrix = sparse_mat + sparse_mat.T + scipy.sparse.eye(
                                                        sparse_mat.shape[0])
    if not sparse_mode:
        similarity_matrix = similarity_matrix.toarray()

    return similarity_matrix


def define_clusts(similarity_matrix, threshold=0.053447011367803443):
    """Define clusters given the similarity matrix and the threshold."""
    n, labels = connected_components(similarity_matrix)
    prev_max_clust = 0
    clusters = labels.copy()
    for i in range(n):
        idxs = np.where(labels == i)
        if idxs[0].shape[0] > 1:
            dm = 1. - similarity_matrix[idxs[0]][:, idxs[0]].toarray()
            links = linkage(squareform(dm), 'ward')
            clusters_ = fcluster(links, threshold,
                                 criterion='distance') + prev_max_clust
            clusters[idxs[0]] = clusters_
            prev_max_clust = max(clusters_)
        else:  # connected component contains just 1 element
            prev_max_clust += 1
            clusters[idxs[0]] = prev_max_clust
    return np.array(extra.flatten(clusters))


def define_clones(db_iter, exp_tag='debug', root=None, force_silhouette=False):
    """Run the pipeline of ignet."""
    if not os.path.exists(root):
        if root is None:
            root = 'results_'+exp_tag+extra.get_time()
        os.makedirs(root)
        logging.warn("No root folder supplied, folder {} "
                     "created".format(os.path.abspath(root)))

    similarity_matrix = compute_similarity_matrix(db_iter, sparse_mode=True)

    output_filename = exp_tag
    output_folder = os.path.join(root, output_filename)

    # Create exp folder into the root folder
    os.makedirs(output_folder)

    sm_filename = output_filename + '_similarity_matrix.pkl.tz'
    with gzip.open(os.path.join(output_folder, sm_filename), 'w+') as f:
        pkl.dump(similarity_matrix, f)
    logging.info("Dumped similarity matrix: {}"
                 .format(os.path.join(output_folder, sm_filename)))

    clusters = define_clusts(similarity_matrix)

    cl_filename = output_filename + '_clusters.pkl.tz'
    with gzip.open(os.path.join(output_folder, cl_filename), 'w+') as f:
        pkl.dump(clusters, f)
    logging.info("Dumped clusters: {}"
                 .format(os.path.join(output_folder, cl_filename)))

    clone_dict = {k.id: v for k, v in zip(db_iter, clusters)}
    return output_folder, clone_dict


if __name__ == '__main__':
    print("This file cannot be launched directly. "
          "Please run the script located in `ignet.scripts.ig_run.py` "
          "with an argument, that is the location of the configuration file. ")
