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
# from scipy.cluster.hierarchy import fcluster, linkage
# from scipy.spatial.distance import squareform
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import AffinityPropagation
from sklearn.utils.sparsetools import connected_components

# from icing.core.distances import string_distance
from icing.core.similarity_scores import similarity_score_tripartite as mwi
from icing.externals import AffinityPropagation
from icing.kernel import sum_string_kernel
from icing.models.model import model_matrix
from icing.utils import extra


def sim_function(ig1, ig2, method='jaccard', model='ham',
                 dist_mat=None, tol=3, v_weight=1., j_weight=1.,
                 correction_function=(lambda _: 1), correct=True,
                 sim_score_params=None, ssk_params=None):
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
    if abs(ig1.junction_length - ig2.junction_length) > tol:
        return 0.

    V_genes_ig1 = ig1.getVGene('set')
    V_genes_ig2 = ig2.getVGene('set')
    J_genes_ig1 = ig1.getJGene('set')
    J_genes_ig2 = ig2.getJGene('set')

    ss = mwi(V_genes_ig1, V_genes_ig2, J_genes_ig1, J_genes_ig2, method=method,
             r1=v_weight, r2=j_weight, sim_score_params=sim_score_params)
    if ss > 0.:
        junc1 = extra.junction_re(ig1.junction)
        junc2 = extra.junction_re(ig2.junction)

        # Using alignment plus model
        # norm_by = min(len(junc1), len(junc2))
        # if model == 'hs1f':
        #     norm_by *= 2.08
        # if dist_mat is None:
        #     dist_mat = model_matrix(model)
        # dist = string_distance(junc1, junc2, dist_mat, norm_by, tol=tol)
        # ss *= (1 - dist)

        # Using string kernel
        ss *= sum_string_kernel(
            [junc1, junc2],
            verbose=False,
            normalize=1, return_float=1, **ssk_params)

    if ss > 0 and correct:
        correction = correction_function(np.mean((ig1.mut, ig2.mut)))
        # correction = min(correction_function(ig1.mut),
        #                  correction_function(ig2.mut))
        # ss = 1 - ((1 - ss) * max(correction, 0))
        # ss = 1 - ((1 - ss) * correction)
        ss *= np.clip(correction, 0, 1)
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
        for v in ig.getVGene('set') or ():
            r_index.setdefault(v, []).append(i)
        for j in ig.getJGene('set') or ():
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
    procs = []

    try:
        queue = manager.list()
        for idx in range(nprocs):
            p = mp.Process(target=_get_v_j_padding,
                           args=(lock, queue, idx, nprocs, records, n))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    except:
        extra.term_processes(procs)

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
            for k in range(int(length * (length - 1) / 2)):
                j = k % (length - 1) + 1
                i = int(k / (length - 1))
                if i >= j:
                    i = length - i - 1
                    j = length - j
                i = v[i]
                j = v[j]
                indicator_matrix[i, j] = similarity_function(
                    records[i], records[j])
    rows, cols, data = map(list, scipy.sparse.find(indicator_matrix))
    return data, rows, cols


# def _similar_elements_job(idx, queue, reverse_index, nprocs):
#     key_list = list(reverse_index)
#     row_local, col_local = np.empty(0, dtype=int), np.empty(0, dtype=int)
#     m = len(key_list)
#     for ii in range(idx, m, nprocs):
#         v = reverse_index[key_list[ii]]
#         length = len(v)
#         if length > 1:
#             combinations = int(length * (length - 1) / 2.)
#             rows = np.empty(combinations, dtype=int)
#             cols = np.empty(combinations, dtype=int)
#             for k in range(combinations):
#                 j = k % (length - 1) + 1
#                 i = int(k / (length - 1))
#                 if i >= j:
#                     i = length - i - 1
#                     j = length - j
#                 i = v[i]
#                 j = v[j]
#                 rows[k] = i
#                 cols[k] = j
#             # row_local = np.hstack((row_local, rows))
#             # col_local = np.hstack((col_local, cols))
#             queue.put((rows, cols))

def _similar_elements_job(value, queue, nprocs):
    v = value
    length = len(v)
    if length > 1:
        combinations = int(length * (length - 1) / 2.)
        rows = np.empty(combinations, dtype=int)
        cols = np.empty(combinations, dtype=int)
        for k in range(combinations):
            j = k % (length - 1) + 1
            i = int(k / (length - 1))
            if i >= j:
                i = length - i - 1
                j = length - j
            i = v[i]
            j = v[j]
            rows[k] = i
            cols[k] = j
        # row_local = np.hstack((row_local, rows))
        # col_local = np.hstack((col_local, cols))
        queue.put((rows, cols))


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
                                            similarity_function)[1:]
    nprocs = min(mp.cpu_count(), n) if nprocs == -1 else nprocs
    manager = mp.Manager()
    queue = manager.Queue()
    job = partial(
        _similar_elements_job, nprocs=nprocs, queue=queue)
    pool = mp.Pool(processes=nprocs)
    rows = np.empty(0, dtype=int)
    cols = np.empty(0, dtype=int)
    gen = (v for v in reverse_index.itervalues())
    from itertools import islice
    class MyIterator(object):
        def __init__(self, iterable):
            self._iterable = iter(iterable)
            self._exhausted = False
            self._cache_next_item()
        def _cache_next_item(self):
            try:
                self._next_item = next(self._iterable)
            except StopIteration:
                self._exhausted = True
        def __iter__(self):
            return self
        def next(self):
            if self._exhausted:
                raise StopIteration
            next_item = self._next_item
            self._cache_next_item()
            return next_item
        def __nonzero__(self):
            return not self._exhausted
        def ended(self):
            return self._exhausted

    g1 = MyIterator(gen)
    while True:
        # pool.map(job, range(nprocs))
        pool.map(job, islice(g1, 5000))
        if queue.empty() and g1.ended():
            break
        while not queue.empty():
            r, c = queue.get()
            rows = np.hstack((rows, r))
            cols = np.hstack((cols, c))
        if g1.ended():
            break
    return rows, cols

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
    procs = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(data, rows, cols, n, records, idx, nprocs,
                                 similarity_function))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        extra.term_processes(procs, 'Exit signal received\n')
    except BaseException as msg:
        extra.term_processes(procs, 'ERROR: %s\n' % msg)

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
    sim_func_args.setdefault('model', default_model)
    sim_func_args.setdefault('dist_mat', model_matrix(default_model))
    sim_func_args.setdefault('tol', 3)
    sim_func_args.setdefault(
        'ssk_params', {'min_kn': 1, 'max_kn': 8, 'lamda': .75})

    dd = inverse_index(igs)
    if sim_func_args.setdefault('method', 'jaccard') \
            in ('pcc', 'hypergeometric'):
        sim_func_args['sim_score_params'] = {
            'nV': len([x for x in dd if 'V' in x]),
            'nJ': len([x for x in dd if 'J' in x])
        }
    logging.info("Similarity function parameters: %s", sim_func_args)
    similarity_function = partial(sim_function, **sim_func_args)

    logging.info("Start similar_elements function ...")
    rows, cols = similar_elements(dd, igs, n, similarity_function)

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

    ap = AffinityPropagation(affinity='precomputed', max_iter=max_iter,
                             preference='median')
    for i in range(n):
        idxs = np.where(labels == i)[0]
        if idxs.shape[0] > 1:
            sm = similarity_matrix[idxs][:, idxs]

            # Hierarchical clustering
            # if method == 'hierarchical':
            #     links = linkage(1. - sm.toarray(), method='ward')
            #     clusters_ = fcluster(links, 1-threshold, 'distance')

            # DBSCAN
            # if method == 'dbscan':
            #     db = ap.fit(1. - sm.toarray())
            #     # Number of clusters in labels, ignoring noise if present.
            #     clusters_ = db.labels_ + 1
            #     # n_clusters_ = len(set(clusters_)) - int(0 in clusters_)

            # AffinityPropagation
            db = ap.fit(sm)
            clusters_ = db.labels_ + 1

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
                  sim_func_args=None, threshold=0.05, db_file=None):
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
    n_clones = np.max(clusters) - np.min(clusters) + 1
    logging.critical("Number of clones: %i, threshold %.3f", n_clones,
                     threshold)
    with open(os.path.join(output_folder, 'summary.txt'), 'w') as f:
        f.write("filename: %s\n" % db_file)
        f.write("clones: %i\n" % n_clones)

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
