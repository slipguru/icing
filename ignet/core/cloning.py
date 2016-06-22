#!/usr/bin/env python
"""Assign Ig sequences into clones."""

import os
import sys
import imp
import multiprocessing as mp
import numpy as np
import scipy
import joblib as jl

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.utils.sparsetools import connected_components

from .distances import junction_distance, string_distance
from .similarity_scores import similarity_score_tripartite as mwi
from .. import parallel_distance
from ..utils import io
from ..utils import utils
from ..plotting import silhouette
from ..externals import DbCore

__author__ = 'Federico Tomasi'


def neg_exp(x, a, c, d):
    return a * np.exp(-c * x) + d


def alpha_mut(ig1, ig2, fn='../models/negexp_pars.npy'):
    '''Coefficient to balance distance according to mutation levels'''
    try:
        popt = np.load(fn)
        return neg_exp(np.max((ig1.mut, ig2.mut)), *popt)
    except:
        return np.exp(-np.max((ig1.mut, ig2.mut)) / 35.)


def dist_function(ig1, ig2, method='jaccard', model='ham', dist_mat=None, tol=3):
    # nmer = 5 if model == 'hs5f' else 1
    if ig1.junction_length - ig2.junction_length > tol:
        return 0.

    if not dist_mat:
        dist_mat = DbCore.getModelMatrix(model)

    V_genes_ig1 = ig1.getVGene('set')
    V_genes_ig2 = ig2.getVGene('set')
    J_genes_ig1 = ig1.getJGene('set')
    J_genes_ig2 = ig2.getJGene('set')

    ss = mwi(V_genes_ig1, V_genes_ig2, J_genes_ig1, J_genes_ig2, method=method)
    if ss > 0.:
        # print V_genes_ig1, V_genes_ig2, J_genes_ig1, J_genes_ig2
        junc1 = utils.junction_re(ig1.junction)
        junc2 = utils.junction_re(ig2.junction)
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

def distance_matrix(config_file, sparse_mode=True):
    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('config', config_path)

    # Load input file
    allowed_subsets = ['n', 'naive']
    allowed_mut = (0, 0)
    # db_iter = io.read_db(config.db_file)
    db_iter = io.read_db(config.db_file, filt=(lambda x: x.subset.lower() in allowed_subsets and allowed_mut[0] <= x.mut <= allowed_mut[1]))
    # igs = np.array(filter(lambda x: x.subset.lower() in allowed_subsets and allowed_mut[0] <= x.mut <= allowed_mut[1], db_iter))
    # igs = np.array([x for x in db_iter])
    #
    # X = parallel_distance.distance_matrix_parallel(igs, d_func, sparse_mode=sparse_mode)
    # # np.savetxt("dist_matrix", dist_matrix, fmt="%.2f", delimiter=',')
    # return X
    # print 'end loading'
    igs = list(db_iter)
    print len(igs)
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
    for i_igs in utils.split_list(list(enumerate(igs)), n_proc):
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
    data = jl.Parallel(n_jobs=-1) \
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
    S_ = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(ndim, ndim))
    similarity_matrix = S_ + S_.T + scipy.sparse.eye(S_.shape[0])
    return similarity_matrix


def define_clusts(similarity_matrix, threshold=0.053447011367803443):
    """Define clusters given the similarity matrix and the threshold."""
    n, labels = connected_components(similarity_matrix)
    prev_max_clust = 0
    clusters = []
    for i in range(n):
        idxs = np.where(labels == i)
        if idxs[0].shape[0] > 1:
            D_ = 1. - similarity_matrix[idxs[0]][:, idxs[0]].toarray()
            links = linkage(squareform(D_), 'ward')
            clusters_ = fcluster(links, threshold,
                                 criterion='distance') + prev_max_clust
            clusters.append(clusters_)
            prev_max_clust = max(clusters_)
        else:  # connected component contains just 1 element
            prev_max_clust += 1
            clusters.append(prev_max_clust)
    return np.array(utils.flatten(clusters))


def sil_score(similarity_matrix, clusters):
    # links = linkage(squareform(1. - similarity_matrix.toarray()), 'ward')
    # clusters = fcluster(links, 0.053447011367803443, criterion='distance')
    silhouette.compute_silhouette_score(1. - similarity_matrix.toarray(), clusters, max(clusters))


def run(config_file):
    similarity_matrix = distance_matrix(config_file)
    clusters = define_clusts(similarity_matrix)
    sil_score(similarity_matrix, clusters)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("USAGE: cloning.py <CONFIG_FILE>")
        sys.exit(-1)

    run(sys.argv[1])
