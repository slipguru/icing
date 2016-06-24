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


def getVJ(d, i_igs):
    for i, ig in i_igs:
        for v in ig.getVGene('set'):
            # d.setdefault(v,[]).append((i,ig))
            d[v] = d.get(v, []) + [(i, ig)]
        for j in ig.getJGene('set'):
            # d.setdefault(j,[]).append((i,ig))
            d[j] = d.get(j, []) + [(i, ig)]


def distance_matrix(db_iter, sparse_mode=True):
    # X = parallel_distance.distance_matrix_parallel(igs, d_func, sparse_mode=sparse_mode)
    # # np.savetxt("dist_matrix", dist_matrix, fmt="%.2f", delimiter=',')
    # return X
    # print 'end loading'
    igs = list(db_iter)
    print("DEBUG # len igs read: ", len(igs))
    # r_index = dict()
    # for i, ig in enumerate(igs):
    #     igs.append(ig)
    #     for v in ig.getVGene('set'): r_index.setdefault(v,[]).append((i,ig))
    #     for j in ig.getJGene('set'): r_index.setdefault(j,[]).append((i,ig))
    manager = mp.Manager()
    r_index = manager.dict()
    ndim = len(igs)
    n_proc = min(ndim, mp.cpu_count())
    ps = []
    for i_igs in extra.split_list(list(enumerate(igs)), n_proc):
        p = mp.Process(target=getVJ, args=(r_index, i_igs))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

    # print 'end inverse index'
    indicator_matrix = scipy.sparse.lil_matrix((ndim, ndim), dtype=np.dtype('bool'))
    r_index = dict(r_index)
    for k, v in r_index.iteritems():
        length = len(v)
        if length > 1:
            for i in range(length):
                for j in range(i+1, length):
                    indicator_matrix[v[i][0], v[j][0]] = True
    rows, cols, _ = map(list, scipy.sparse.find(indicator_matrix))
    data = jl.Parallel(n_jobs=-1)\
        (jl.delayed(d_func)(igs[i], igs[j]) for i, j in zip(rows, cols))
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
    sparse_mat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(ndim, ndim))
    similarity_matrix = sparse_mat + sparse_mat.T + scipy.sparse.eye(sparse_mat.shape[0])
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


def sil_score(similarity_matrix, clusters):
    """Compute the silhouette score given the similarity matrix."""
    # links = linkage(squareform(1. - similarity_matrix.toarray()), 'ward')
    # clusters = fcluster(links, 0.053447011367803443, criterion='distance')
    if similarity_matrix.shape[0] > 8000:
        sys.stderr.write("Warning: you are about to allocate a 500MB "
                         "matrix in memory.\n")
    silhouette.plot_clusters_silhouette(1. - similarity_matrix.toarray(),
                                        clusters, max(clusters))


def define_clones(db_iter, exp_tag='debug', root=None):
    """Run the pipeline of ignet."""
    if not os.path.exists(root):
        if root is None:
            root = 'results_'+exp_tag+extra.get_time()
        os.makedirs(root)
        logging.warn("No root folder supplied, folder {} "
                     "created".format(os.path.abspath(root)))

    similarity_matrix = distance_matrix(db_iter)

    output_filename = exp_tag
    output_folder = os.path.join(root, output_filename)

    # Create exp folder into the root folder
    os.makedirs(output_folder)

    with gzip.open(os.path.join(output_folder, output_filename +
                                '_similarity_matrix.pkl.tz'), 'w+') as f:
        pkl.dump(similarity_matrix, f)
    logging.info("Dumped similarity matrix: {}"
                 .format(os.path.join(output_folder, output_filename+'.pkl.tz')))

    clusters = define_clusts(similarity_matrix)

    if similarity_matrix.shape[0] < 8000:
        sil_score(similarity_matrix, clusters)
    else:
        sys.stderr.write("Silhouette analysis is not performed due to the "
                         "matrix dimensions.\n")

    clone_dict = {k.id: v for k, v in zip(db_iter, clusters)}
    print(clone_dict)
    return output_folder, clone_dict


if __name__ == '__main__':
    print("This file cannot be launched directly. "
          "Please run the script located in `ignet.scripts.ig_run.py` "
          "with an argument, that is the location of the configuration file. ")
