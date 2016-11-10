#!/usr/bin/env python
"""Assign Ig sequences into clones.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
from __future__ import print_function

import cPickle as pkl
import gzip
import logging
import multiprocessing as mp
import numpy as np
import os
import scipy

from collections import defaultdict
from functools import partial
from pysapc import SAP  # Sparse Affinity Propagation
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.utils.sparsetools import connected_components

from icing.core.distances import string_distance
from icing.core.similarity_scores import similarity_score_tripartite as mwi
from icing.models.model import model_matrix
from icing.utils import extra
from string_kernel.core.src.sum_string_kernel import sum_string_kernel


def alpha_mut(ig1, ig2, fn='models/negexp_pars.npy'):
    """Coefficient to balance distance according to mutation levels."""
    try:
        params_folder = os.path.abspath(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), os.pardir))
        popt = np.load(os.path.join(params_folder, fn))
        # return _neg_exp(np.max((ig1.mut, ig2.mut)), *popt)
        return popt
    except Exception:
        logging.error("Correction coefficient file not found. "
                      "No correction can be applied for mutation.")
        # return np.exp(-np.max((ig1.mut, ig2.mut)) / 35.)
        return (1, 0, 0)


def sim_function(ig1, ig2, method='jaccard', model='ham',
                 dist_mat=None, tol=3, v_weight=1., j_weight=1.,
                 correction_function=(lambda _: 1), correct=True,
                 sim_score_params=None):
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

    ss = mwi(V_genes_ig1, V_genes_ig2, J_genes_ig1, J_genes_ig2, method=method,
             r1=v_weight, r2=j_weight, sim_score_params=sim_score_params)
    if ss > 0.:
        # print V_genes_ig1, V_genes_ig2, J_genes_ig1, J_genes_ig2
        junc1 = extra.junction_re(ig1.junction)
        junc2 = extra.junction_re(ig2.junction)
        # norm_by = min(len(junc1), len(junc2))
        # if model == 'hs1f':
        #     norm_by *= 2.08
        # dist = string_distance(junc1, junc2, dist_mat, norm_by, tol=tol)
        # ss *= (1 - dist)

        min_kn = 1
        max_kn = 5
        lamda = .75
        normalize = 1
        ss *= sum_string_kernel(
            [junc1, junc2],
            min_kn=min_kn, max_kn=max_kn, lamda=lamda, verbose=False,
            normalize=normalize, return_float=1)

    if ss > 0 and correct:
        correction = correction_function(np.mean((ig1.mut, ig2.mut)))
        # ss = 1 - ((1 - ss) * max(correction, 0))
        # ss = 1 - ((1 - ss) * correction)
        ss *= correction
    # return min(max(ss, 0), 1)
    return max(ss, 0)


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
    r_index = dict()
    for i, ig in enumerate(records):
        for v in ig.getVGene('set'):
            r_index.setdefault(v, []).append(i)
        for j in ig.getJGene('set'):
            r_index.setdefault(j, []).append(i)

    # r_index = defaultdict(list)
    # for i, ig in enumerate(list(records)):
    #     for v in ig.getVGene('set'):
    #         r_index[v].append(i)
    #     for j in ig.getJGene('set'):
    #         r_index[j].append(i)

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
    def _get_v_j_padding(lock, queue, idx, nprocs, igs_arr, n):
        local_dict = defaultdict(list)
        for i in range(idx, n, nprocs):
            ig = igs_arr[i]
            for _ in ig.getVGene('set'):
                local_dict[_].append(i)
            for _ in ig.getJGene('set'):
                local_dict[_].append(i)
        with lock:
            queue.append(dict(local_dict))

    n = len(records)
    manager = mp.Manager()
    lock = mp.Lock()
    nprocs = min(n, mp.cpu_count())
    ps = []

    try:
        queue = manager.list()
        for idx in range(nprocs):
            p = mp.Process(target=_get_v_j_padding,
                           args=(lock, queue, idx, nprocs, records, n))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    except:
        extra.term_processes(ps)

    reverse_index = {}
    for dictionary in queue:
        reverse_index = dict(reverse_index.items() + dictionary.items() +
                             [(k, list(set(reverse_index[k] + dictionary[k])))
                              for k in set(dictionary) & set(reverse_index)])

    return reverse_index


def _similar_elements_sequential(reverse_index, records, n,
                                 similarity_function):
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
                indicator_matrix[i, j] = similarity_function(
                    records[i], records[j])
                # If the distance function is not symmetric,
                # then turn the following on:
                # indicator_matrix[v[j][0], v[i][0]] = True
    rows, cols, data = map(list, scipy.sparse.find(indicator_matrix))
    return data, rows, cols


def _similar_elements_job(
        reverse_index, records, n, idx, nprocs, data, rows, cols, sim_func):
    key_list = list(reverse_index)
    m = len(key_list)
    for ii in range(idx, m, nprocs):
        v = reverse_index[key_list[ii]]
        length = len(v)
        if length > 1:
            for k in (range(int(length * (length - 1) / 2))):
                j = k % (length - 1) + 1
                i = int(k / (length - 1))
                if i >= j:
                    i = length - i - 1
                    j = length - j
                i = v[i]
                j = v[j]
                c_idx = n*(n-1)/2 - (n - i) * (n - i - 1) / 2 + j - i - 1
                rows[c_idx] = int(i)
                cols[c_idx] = int(j)
                data[c_idx] = 1  # sim_func(records[i], records[j])


def similar_elements(reverse_index, records, n, similarity_function,
                     nprocs=-1):
    """Return the sparse similarity matrix in form of data, rows and cols.

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
    nprocs : int, optional, default: -1
        Number of processes to use.

    Returns
    -------
    _ : set
        Set of pairs of elements on which further calculate the similarity.
    """
    if nprocs == 1:
        return _similar_elements_sequential(reverse_index, records, n,
                                            similarity_function)
    nprocs = min(mp.cpu_count(), n) if nprocs == -1 else nprocs
    c_length = int(n * (n - 1) / 2)
    data = mp.Array('d', [0] * c_length)
    rows = mp.Array('d', [0] * c_length)
    cols = mp.Array('d', [0] * c_length)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_similar_elements_job,
                           args=(reverse_index, records, n, idx, nprocs,
                                 data, rows, cols, similarity_function))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra.term_processes(ps, 'Exit signal received\n')
    except Exception as e:
        extra.term_processes(ps, 'ERROR: %s\n' % e)

    data = np.array(data, dtype=float)
    idx = data > 0
    data = data[idx]
    rows = np.array(rows, dtype=int)[idx]
    cols = np.array(cols, dtype=int)[idx]
    return data, rows, cols


def indicator_to_similarity(rows, cols, records, similarity_function):
    """Given the position on a sparse matrix, compute the similarity.

    Parameters:
    -----------
    rows, cols : array_like
        Positions of records to calculate similarities.
    records : array_like
        Records to use for the ocmputation.
    similarity_function : function
        Function to calculate similarities.

    Returns:
    --------
    data : multiprocessing.array
        Array of length len(rows) which contains similarities among records
        as specified by rows and cols.
    """
    def _internal(data, rows, cols, n, records,
                  idx, nprocs, similarity_function):
        for i in range(idx, n, nprocs):
            data[i] = similarity_function(records[rows[i]], records[cols[i]])

    n = len(rows)
    nprocs = min(mp.cpu_count(), n)
    data = mp.Array('d', [0.] * n)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(data, rows, cols, n, records, idx, nprocs,
                                 similarity_function))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra.term_processes(ps, 'Exit signal received\n')
    except Exception as e:
        extra.term_processes(ps, 'ERROR: %s\n' % e)

    return data


def compute_similarity_matrix(db_iter, sparse_mode=True, **sim_func_args):
    """Compute the similarity matrix from a database iterator.

    Parameters
    ----------
    db_iter : generator
        Records loaded by the script `ici_run.py`.
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
    ``parallel_distance.distance_matrix_parallel(db_iter, d_func, sparse_mode)``
    However, this method is really inefficient, so the matrix is computed
    between chosen couples of values.
    """
    igs = list(db_iter)
    n = len(igs)

    default_model = 'ham'
    default_method = 'jaccard'
    sim_func_args.setdefault('model', default_model)
    sim_func_args.setdefault('method', default_method)
    sim_func_args.setdefault('tol', 3)
    dist_mat = sim_func_args.get('dist_mat', None)
    if dist_mat is None:
        dist_mat = sim_func_args.setdefault('dist_mat',
                                            model_matrix(default_model))

    dd = inverse_index(igs)
    if sim_func_args.get('method', None) in ('pcc', 'hypergeometric'):
        sim_func_args['sim_score_params'] = {
            'nV': len([x for x in dd if 'V' in x]),
            'nJ': len([x for x in dd if 'J' in x])
        }
    logging.info("Similarity function parameters: %s", sim_func_args)
    similarity_function = partial(sim_function, **sim_func_args)

    logging.info("Start similar_elements function ...")
    _, rows, cols = similar_elements(dd, igs, n, similarity_function)

    logging.info("Start parallel_sim_matrix function ...")
    data = indicator_to_similarity(rows, cols, igs, similarity_function)

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

    sparse_mat = scipy.sparse.csr_matrix((data, (rows, cols)),
                                         shape=(n, n))
    similarity_matrix = sparse_mat + sparse_mat.T + scipy.sparse.eye(
        sparse_mat.shape[0])
    if not sparse_mode:
        similarity_matrix = similarity_matrix.toarray()

    return similarity_matrix


def define_clusts(similarity_matrix, threshold=0.05, max_iter=200):
    """Define clusters given the similarity matrix and the threshold."""
    n, labels = connected_components(similarity_matrix, directed=False)
    prev_max_clust = 0
    clusters = labels.copy()

    def clusters_dict(a):
        d = {}
        count = 0
        for e in a:
            if d.get(e, None) is None:
                d[e] = count
                count += 1
        return [d[e] for e in a]

    for i in range(n):
        idxs = np.where(labels == i)[0]
        if idxs.shape[0] > 1:
            # dm = 1. - similarity_matrix[idxs][:, idxs].toarray()

            # # Hierarchical clustering
            # links = linkage((dm), method='ward')
            # clusters_ = fcluster(links, threshold, 'distance')

            # DBSCAN
            # db = DBSCAN(eps=threshold, min_samples=1,
            #             metric='precomputed').fit(dm)
            # # Number of clusters in labels, ignoring noise if present.
            # n_clusters_ = len(set(clusters_)) - (1 if -1 in clusters_ else 0)
            # print('Estimated number of clusters by DBSCAN: %d' % n_clusters_)

            # # AffinityPropagation
            db = AffinityPropagation(affinity='precomputed',
                                     max_iter=max_iter) \
                .fit(similarity_matrix[idxs][:, idxs].toarray())
            clusters_ = np.array(db.labels_, dtype=int) + 1

            # # SparseAffinityPropagation
            # db = SAP(verboseIter=0, damping=.5) \
            #     .fit(similarity_matrix[idxs][:, idxs], 'median')
            # clusters_ = np.array(clusters_dict(db.exemplars_), dtype=int) + 1
            clusters_ += prev_max_clust
            clusters[idxs] = clusters_
            prev_max_clust = max(clusters_)
        else:  # connected component contains just 1 element
            prev_max_clust += 1
            clusters[idxs] = prev_max_clust
    return np.array(extra.flatten(clusters))

# def define_clusts_(similarity_matrix, threshold):
#     """Define clusters given the similarity matrix and the threshold."""
#     dm = 1. - similarity_matrix.toarray()
#     links = linkage((dm), method='ward')
#     clusters = fcluster(links, threshold,   'distance')
#     return np.array(extra.flatten(clusters))


def define_clones(db_iter, exp_tag='debug', root=None,
                  sim_func_args=None, threshold=0.05):
    """Run the pipeline of icing."""
    if not os.path.exists(root):
        if root is None:
            root = 'results_' + exp_tag + extra.get_time()
        os.makedirs(root)
        logging.warn("No root folder supplied, folder %s "
                     "created", os.path.abspath(root))

    similarity_matrix = compute_similarity_matrix(db_iter, sparse_mode=True,
                                                  **sim_func_args)

    output_filename = exp_tag
    output_folder = os.path.join(root, output_filename)

    # Create exp folder into the root folder
    os.makedirs(output_folder)

    sm_filename = output_filename + '_similarity_matrix.pkl.tz'
    with gzip.open(os.path.join(output_folder, sm_filename), 'w+') as f:
        pkl.dump(similarity_matrix, f)
    logging.info("Dumped similarity matrix: %s",
                 os.path.join(output_folder, sm_filename))

    logging.info("Start define_clusts function ...")
    clusters = define_clusts(similarity_matrix, threshold=threshold)
    logging.critical("Number of clones: %i, threshold %.3f",
                     np.max(clusters) - np.min(clusters) + 1, threshold)

    cl_filename = output_filename + '_clusters.pkl.tz'
    with gzip.open(os.path.join(output_folder, cl_filename), 'w+') as f:
        pkl.dump([clusters, threshold], f)
    logging.info("Dumped clusters and threshold: %s",
                 os.path.join(output_folder, cl_filename))

    clone_dict = {k.id: v for k, v in zip(db_iter, clusters)}
    return output_folder, clone_dict


if __name__ == '__main__':
    print("This file cannot be launched directly. "
          "Please run the script located in `icing/scripts/ici_run.py` "
          "with its configuration file. ")
