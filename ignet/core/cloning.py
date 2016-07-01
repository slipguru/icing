#!/usr/bin/env python
"""Assign Ig sequences into clones."""
from __future__ import print_function

import os
import sys
import imp
import time
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


def sim_function(ig1, ig2, method='jaccard', model='ham',
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

    if dist_mat is None:
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


def inverse_index(records):
    """Compute a inverse index given records, based on their V and J genes.

    Parameters
    ----------
    records : list of externals.DbCore.IgRecord
        Records must have the getVGene and getJGene methods.

    Returns
    -------
    reverse_index : dict
        Reverse index. For each key (a string which represent a gene), it has
        the list of indices of records which contains that gene.
        Example: reverse_index = {'IGHV3' : [0,3,5] ...}
    """
    from collections import defaultdict
    r_index = defaultdict(list)
    for i, ig in enumerate(list(records)):
        for _ in ig.getVGene('set'):
            r_index[_].append(i)
        for _ in ig.getJGene('set'):
            r_index[_].append(i)

    return r_index


def inverse_index_parallel(records):
    """Compute a inverse index given records, based on their V and J genes.

    It computes in a parallel way.

    Parameters
    ----------
    records : list of externals.DbCore.IgRecord
        Records must have the getVGene and getJGene methods.

    Returns
    -------
    reverse_index : dict
        Reverse index. For each key (a string which represent a gene), it has
        the list of indices of records which contains that gene.
        Example: reverse_index = {'IGHV3' : [0,3,5] ...}
    """
    def _get_v_j(d, i_igs):
        for i, ig in i_igs:
            for v in ig.getVGene('set'):
                # d.setdefault(v,[]).append((i,ig))
                d[v] = d.get(v, []) + [(i, )]
            for j in ig.getJGene('set'):
                # d.setdefault(j,[]).append((i,ig))
                d[j] = d.get(j, []) + [(i, )]

    def _get_v_j_queue(d, i_igs):
        for i, ig in i_igs:
            for v in ig.getVGene('set'):
                # d.setdefault(v,[]).append((i,ig))
                d.put((v, (i,)))
            for j in ig.getJGene('set'):
                d.put((j, (i,)))

    from collections import defaultdict

    def _get_v_j_queue2(lock, q, i_igs):
        local_dict = defaultdict(list)
        for i, ig in i_igs:
            for _ in ig.getVGene('set'):
                local_dict[_].append((i,))
                # if _ == 'IGHV6-1':
                #     print("ciao", i)
            for _ in ig.getJGene('set'):
                local_dict[_].append((i,))
        with lock:
            # q.put(dict(local_dict))
            q.append(local_dict)

    def _get_v_j_padding(d, idx, nprocs, igs_arr, n):
        for i in range(idx, n, nprocs):
            ig = igs_arr[i]
            for v in ig.getVGene('set'):
                # d.setdefault(v,[]).append((i,ig))
                d[v] = d.get(v, []) + [(i, )]
            for j in ig.getJGene('set'):
                # d.setdefault(j,[]).append((i,ig))
                d[j] = d.get(j, []) + [(i, )]

    def _get_v_j_padding2(lock, q, idx, nprocs, igs_arr, n):
        local_dict = defaultdict(list)
        for i in range(idx, n, nprocs):
            ig = igs_arr[i]
            for _ in ig.getVGene('set'):
                local_dict[_].append((i,))
                # if _ == 'IGHV6-1':
                #     print("ciao", i)
            for _ in ig.getJGene('set'): local_dict[_].append((i,))
        with lock:
            q.append(dict(local_dict))

    def combine_dicts(a, b):
        return dict(a.items() + b.items() + [(k, list(set(a[k] + b[k]))) for k in set(b) & set(a)])

    n = len(records)
    logging.info("{} Igs read.".format(n))

    manager = mp.Manager()
    # r_index = manager.dict()
    lock = mp.Lock()

    nprocs = min(n, mp.cpu_count())
    ps = []

    queue = manager.list()
    for idx in range(nprocs):
        p = mp.Process(target=_get_v_j_padding2,
                       args=(lock, queue, idx, nprocs, records, n))
        p.start()
        ps.append(p)
    for i, p in enumerate(ps):
        p.join()
        print("collected", i)

    dd = {}
    for d__ in queue:
        dd = combine_dicts(dd, d__)
    return dd


def similar_elements(reverse_index, records, n, distance_function):
    """Return a set of elements on which to compute similarity.

    Parameters
    ----------
    reverse_index : dict
        Reverse index. For each key (a string which represent a gene), it has
        the list of indices of records which contains that gene.
        Example: reverse_index = {'IGHV3' : [0,3,5] ...}
    n : int
        Number of records.

    Returns
    -------
    _ : set
        Set of pairs of elements on which further calculate the similarity.
    """
    # To avoid memory problems, use a sparse matrix.
    # lil_matrix is good to update
    indicator_matrix = scipy.sparse.lil_matrix((n, n), dtype=float)
    #    dtype=np.dtype('bool'))

    for k, v in reverse_index.iteritems():
        length = len(v)
        if length > 1:
            for k in (range(int(length*(length-1)/2))):
                j = k % (length-1)+1
                i = int(k/(length-1))
                if i >= j:
                    i = length-i-1
                    j = length-j
                i = v[i]
                j = v[j]
                indicator_matrix[i, j] = distance_function(records[i], records[j])
                # If the distance function is not symmetric,
                # then turn the following on:
                # indicator_matrix[v[j][0], v[i][0]] = True
    rows, cols, data = map(list, scipy.sparse.find(indicator_matrix))
    return data, rows, cols


def similar_elements_parallel(reverse_index, records, n, similarity_function):
    """Return the sparse similarity matrix in form of data, rows and cols.

    Differently from similar_elements(), perform the calculation in a parallel
    way.

    Parameters
    ----------
    reverse_index : dict
        Reverse index. For each key (a string which represent a gene), it has
        the list of indices of records which contains that gene.
        Example: reverse_index = {'IGHV3' : [0,3,5] ...}
    records : list of externals.DbCore.IgRecord
        List of records to analyse.
    n : int
        Number of records.
    similarity_function : function
        Function which computes the similarity between two records.

    Returns
    -------
    _ : set
        Set of pairs of elements on which further calculate the similarity.
    """
    def _internal(reverse_index, records, n, idx, nprocs, data, rows, cols,
                  distance_function):
        key_list = list(reverse_index)
        m = len(key_list)
        for ii in range(idx, m, nprocs):
            v = reverse_index[key_list[ii]]
            length = len(v)
            if length > 1:
                for k in (range(int(length*(length-1)/2))):
                    j = k % (length-1)+1
                    i = int(k/(length-1))
                    if i >= j:
                        i = length-i-1
                        j = length-j
                    i = v[i]
                    j = v[j]
                    c_idx = n*(n-1)/2 - (n-i)*(n-i-1)/2 + j - i - 1
                    rows[c_idx] = int(i)
                    cols[c_idx] = int(j)
                    data[c_idx] = 1#similarity_function(records[i], records[j])

    nprocs = min(mp.cpu_count(), n)
    c_length = int(n*(n-1)/2)
    data = mp.Array('d', [0]*c_length)
    rows = mp.Array('d', [0]*c_length)
    cols = mp.Array('d', [0]*c_length)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(reverse_index, records, n, idx, nprocs,
                                 data, rows, cols, similarity_function))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra._terminate(ps, 'Exit signal received\n')
    except Exception as e:
        extra._terminate(ps, 'ERROR: %s\n' % e)
    except:
        extra._terminate(ps, 'ERROR: Exiting with unknown exception\n')

    data = np.array(data, dtype=float)
    idx = data > 0
    data = data[idx]
    rows = np.array(rows, dtype=int)[idx]
    cols = np.array(cols, dtype=int)[idx]
    return data, rows, cols


def parallel_sim_matrix(rows, cols, records, n, similarity_function):
    def _internal(data, rows, cols, records, idx, nprocs, similarity_function):
        for i in range(idx, n, nprocs):
            data[i] = similarity_function(records[rows[i]], records[cols[i]])

    n = len(rows)
    nprocs = min(mp.cpu_count(), n)
    data = mp.Array('d', [0.]*n)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(data, rows, cols, records, idx, nprocs,
                                 similarity_function))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        _terminate(ps, 'Exit signal received\n')
    except Exception as e:
        _terminate(ps, 'ERROR: %s\n' % e)
    except:
        _terminate(ps, 'ERROR: Exiting with unknown exception\n')

    return data


def compute_similarity_matrix(db_iter, sparse_mode=True, **sim_func_args):
    """Compute the similarity matrix from a database iterator.

    Parameters
    ----------
    db_iter : generator
        Records loaded by the script `ig_run.py`.
    sparse_mode : bool, optional, default `True`
        Return a sparse similarity matrix.
    sim_func_args : dict, optional
        Optional parameters for the similarity function.

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
    igs = list(db_iter)
    n = len(igs)

    dd = inverse_index(igs)

    from functools import partial
    default_model = 'ham'
    default_method = 'jaccard'
    sim_func_args.setdefault('model', default_model)
    sim_func_args.setdefault('method', default_method)
    sim_func_args.setdefault('tol', 3)
    dist_mat = sim_func_args.get('dist_mat', None)
    if dist_mat is None:
        dist_mat = sim_func_args.setdefault('dist_mat',
                                            model_matrix(default_model))

    print("Similarity function parameters: ", sim_func_args)
    similarity_function = partial(sim_function, **sim_func_args)

    tic = time.time()
    # data, rows, cols = similar_elements_parallel(dd, igs, n, similarity_function)
    _, rows, cols = similar_elements_parallel(dd, igs, n, similarity_function)
    data = parallel_sim_matrix(rows, cols, igs, n, similarity_function)
    print(time.time()-tic)

    data = np.array(data, dtype=float)
    idx = data > 0
    data = data[idx]
    rows = np.array(rows, dtype=int)[idx]
    cols = np.array(cols, dtype=int)[idx]

    # tic = time.time()
    # data, rows, cols = similar_elements(dd, igs, n, similarity_function)
    # print(time.time()-tic)
    # rows, cols, _ = map(list, scipy.sparse.find(indicator_matrix))
    # data = jl.Parallel(n_jobs=-1)(jl.delayed(d_func)
    #                               (igs[i], igs[j]) for i, j in s2)

    # inefficient:
    # S = set()
    # for k,v in r_index.iteritems():
    #     length = len(v)
    #     if length > 1:
    #         S.update([(v[i], v[j]) for i in range(length) for j in range(i+1, length)])

    # d_func = lambda x, y: dist_function(x, y, 'jaccard', 'ham')
    # d_wrapper = lambda x, y: (d_func(x[1], y[1]), (x[0], y[0]))

    sparse_mat = scipy.sparse.csr_matrix((data, (rows, cols)),
                                          shape=(n, n))
    similarity_matrix = sparse_mat + sparse_mat.T + scipy.sparse.eye(
                                                        sparse_mat.shape[0])
    if not sparse_mode:
        similarity_matrix = similarity_matrix.toarray()

    np.savetxt("simmat4.csv", similarity_matrix.toarray(), fmt='%.3f')

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


def define_clones(db_iter, exp_tag='debug', root=None, force_silhouette=False,
                  sim_func_args={}):
    """Run the pipeline of ignet."""
    if not os.path.exists(root):
        if root is None:
            root = 'results_'+exp_tag+extra.get_time()
        os.makedirs(root)
        logging.warn("No root folder supplied, folder {} "
                     "created".format(os.path.abspath(root)))

    similarity_matrix = compute_similarity_matrix(db_iter, sparse_mode=True,
                                                  **sim_func_args)

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
