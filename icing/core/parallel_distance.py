#!/usr/bin/env python
"""Function to compute a user-defined distance function among lists.

Distances can be between one (intra-distance) or two (inter-distances) lists.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import multiprocessing as mp
import numpy as np
import scipy
import scipy.spatial

from itertools import chain, ifilter, combinations, islice
from icing.utils.extra import term_processes, progressbar


def _min(generator, func):
    try:
        return func(generator)
    except ValueError:
        return 0


def dnearest_inter_padding(l1, l2, dist_function, filt=None, func=min):
    """Compute in a parallel way a dist2nearest for two 1-d arrays.

    Use this function with different arrays; if l1 == l2, then the
    results is a 0-array.

    Parameters
    ----------
    l1, l2 : array_like
        1-dimensional arrays. Compute the nearest element of l2 to l1.
    dist_function : function
        Function to use for the distance computation.
    filt : function or None, optional
        Filter based on the result of the distance function.
    func : function, optional, default: min (built-in function)
        Function to apply for selecting the best. Use min for distances,
        max for similarities (consider numpy variants for speed).

    Returns
    -------
    dist2nearest : array_like
        1-D array
    """
    def _internal(l1, l2, n, idx, nprocs, shared_arr, dist_function):
        for i in range(idx, n, nprocs):
            # if i % 100 == 0:
            #     progressbar(i, n)
            shared_arr[i] = _min(
                ifilter(filt, (dist_function(l1[i], el2) for el2 in l2)),
                func)

    n = len(l1)
    nprocs = min(mp.cpu_count(), n)
    shared_array = mp.Array('d', [0.] * n)
    procs = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(l1, l2, n, idx, nprocs, shared_array,
                                 dist_function))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        term_processes(procs, 'Exit signal received\n')
    except BaseException as msg:
        term_processes(procs, 'ERROR: %s\n' % msg)

    # progressbar(n, n)
    return shared_array


def dnearest_intra_padding(l1, dist_function, filt=None, func=min):
    """Compute in a parallel way a dist2nearest for a 1-d arrays.

    For each element in l1, find its closest (without considering itself).

    Parameters
    ----------
    l1 : array_like
        1-dimensional array.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist2nearest : array_like
        1-D array
    """
    def _internal(l1, n, idx, nprocs, shared_arr, dist_function):
        for i in range(idx, n, nprocs):
            # if i % 100 == 0:
            #     progressbar(i, n)
            shared_arr[i] = _min(ifilter(filt, chain(
                (dist_function(l1[i], l1[j]) for j in range(0, i)),
                (dist_function(l1[i], l1[j]) for j in range(i + 1, n))
            )), func)

    n = len(l1)
    nprocs = min(mp.cpu_count(), n)
    shared_array = mp.Array('d', [0.] * n)
    procs = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(l1, n, idx, nprocs, shared_array,
                                 dist_function))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        term_processes(procs, 'Exit signal received\n')
    except BaseException as msg:
        term_processes(procs, 'ERROR: %s\n' % msg)

    # progressbar(n, n)
    return shared_array


def dm_dense_inter_padding(l1, l2, dist_function, condensed=False):
    """Compute in a parallel way a distance matrix for a 1-d array.

    Parameters
    ----------
    l1, l2 : array_like
        1-dimensional arrays. Compute the distance matrix for each couple of
        elements of l1 and l2.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Symmetric NxN distance matrix for each input_array element.
    """
    def _internal(l1, l2, n, idx, nprocs, shared_arr, dist_function):
        for i in range(idx, n, nprocs):
            if i % 100 == 0:
                progressbar(i, n)
            shared_arr[i] = [dist_function(l1[i], el2) for el2 in l2]

    n, m = len(l1), len(l2)
    nprocs = min(mp.cpu_count(), n)
    # index = mp.Value('i', 0)
    # lock = mp.Lock()
    shared_array = np.frombuffer(mp.Array('d', n*m).get_obj()).reshape((n, m))
    procs = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(l1, l2, n, idx, nprocs, shared_array,
                                 dist_function))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        term_processes(procs, 'Exit signal received\n')
    except BaseException as msg:
        term_processes(procs, 'ERROR: %s\n' % msg)

    # progressbar(n,n)
    return shared_array.flatten() if condensed else shared_array


def dm_dense_intra_padding(l1, dist_function, condensed=False):
    """Compute in a parallel way a distance matrix for a 1-d array.

    Parameters
    ----------
    l1, l2 : array_like
        1-dimensional arrays. Compute the distance matrix for each couple of
        elements of l1.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Symmetric NxN distance matrix for each input_array element.
    """
    def _internal(l1, n, idx, nprocs, shared_arr, dist_function):
        for i in range(idx, n, nprocs):
            if i % 100 == 0:
                progressbar(i, n)
            # shared_arr[i, i:] = [dist_function(l1[i], el2) for el2 in l2]
            for j in range(i + 1, n):
                shared_arr[idx, j] = dist_function(l1[i], l1[j])
                # if shared_arr[idx, j] == 0:
                # print l1[i].junction, '\n', l1[j].junction, '\n----------'

    n = len(l1)
    nprocs = min(mp.cpu_count(), n)
    shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n, n))
    procs = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(l1, n, idx, nprocs, shared_array,
                                 dist_function))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
    except (KeyboardInterrupt, SystemExit):
        term_processes(procs, 'Exit signal received\n')
    except BaseException as msg:
        term_processes(procs, 'ERROR: %s\n' % msg)

    # progressbar(n,n)
    dist_matrix = shared_array + shared_array.T
    if condensed:
        dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
    return dist_matrix


def sm_sparse(X, metric, tol):
    """Compute in a parallel way a sim matrix for a 1-d array.

    Parameters
    ----------
    l1 : array_like
        1-dimensional arrays. Compute the distance matrix for each couple of
        elements of l1.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Symmetric NxN distance matrix for each input_array element.
    """
    def _internal(X, metric, iterator, idx, return_queue):
        data = np.empty(0, dtype=float)
        rows = np.empty(0, dtype=int)
        cols = np.empty(0, dtype=int)
        for i, j in iterator:
            res = metric(X[i], X[j])
            if res > 0:  # np.random.ranf(1)[0] / 10:
                data = np.append(data, res)
                rows = np.append(rows, i)
                cols = np.append(cols, j)
                # return_queue.put((res, i, j), False)
        return_queue.put((data, rows, cols), False)
        return_queue.put(None, True)

    n = X.shape[0]
    nprocs = min(mp.cpu_count(), n)
    # allows fast and lighter computation
    iterator = list(
        (i, j) for i, j in combinations(range(X.shape[0]), 2) if
        len(X[i].setV & X[j].setV) > 0 and
        abs(X[i].junction_length - X[j].junction_length) < tol)
    len_it = len(iterator)
    procs = []
    manager = mp.Manager()
    return_queue = manager.Queue()
    data = np.empty(0, dtype=float)
    rows = np.empty(0, dtype=int)
    cols = np.empty(0, dtype=int)
    try:
        for idx in range(nprocs):
            num_elem = int(len_it / nprocs) + 1
            itera = iterator[:num_elem]
            iterator = iterator[num_elem:]
            # itera = list(islice(iterator, int(n * (n - 1) / (2. * nprocs)) + 1))
            p = mp.Process(
                target=_internal,
                args=(X, metric, itera, idx, return_queue))
            p.start()
            procs.append(p)

        count = 0
        while count < nprocs:
            v = return_queue.get(True)
            if v is None:
                count += 1
                continue
            # print("v", v)
            data = np.hstack((data, v[0]))
            rows = np.hstack((rows, v[1]))
            cols = np.hstack((cols, v[2]))
            # data = np.append(data, v[0])
            # rows = np.append(rows, v[1])
            # cols = np.append(cols, v[2])

        for p in procs:
            p.join()
        assert return_queue.empty()

    except (KeyboardInterrupt, SystemExit):
        term_processes(procs, 'Exit signal received\n')
    except BaseException as msg:
        term_processes(procs, 'ERROR: %s\n' % msg)

    return data, rows, cols


def dm_sparse_intra_padding(l1, dist_function, condensed=False):
    """Compute in a parallel way a distance matrix for a 1-d input array.

    Parameters
    ----------
    l1 : array_like
        1-dimensional array for which to compute the distance matrix.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Sparse symmetric NxN distance matrix for each input_array element.
    """
    def _internal(l1, n, idx, nprocs, rows, cols, data, dist_function):
        for i in range(idx, n, nprocs):
            if i % 100 == 0:
                progressbar(i, n)
            # shared_arr[i, i:] = [dist_function(l1[i], el2) for el2 in l2]
            for j in range(i + 1, n):
                # shared_arr[idx, j] = dist_function(l1[i], l1[j])
                _res = dist_function(l1[i], l1[j])
                if _res > 0:
                    c_idx = n*(n-1)/2 - (n-i)*(n-i-1)/2 + j - i - 1
                    data[c_idx] = _res
                    rows[c_idx] = i
                    cols[c_idx] = j

    n = len(l1)
    nprocs = min(mp.cpu_count(), n)
    c_length = int(n * (n - 1) / 2)
    data = mp.Array('d', [0.] * c_length)
    rows = mp.Array('d', [0.] * c_length)
    cols = mp.Array('d', [0.] * c_length)
    procs = []
    try:
        for idx in range(nprocs):
            process = mp.Process(target=_internal,
                           args=(l1, n, idx, nprocs, rows, cols, data,
                                 dist_function))
            process.start()
            procs.append(process)
        for process in procs:
            process.join()
    except (KeyboardInterrupt, SystemExit):
        term_processes(procs, 'Exit signal received\n')
    except BaseException as msg:
        term_processes(procs, 'ERROR: %s\n' % msg)

    data = np.array(data)
    idx = data > 0
    data = data[idx]
    rows = np.array(rows)[idx]
    cols = np.array(cols)[idx]
    # print (data)
    D = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    dist_matrix = D + D.T
    if condensed:
        dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
    progressbar(n, n)
    return dist_matrix


def distance_matrix_parallel(input_array, dist_function, condensed=False,
                             sparse_mode=False):
    """TODO."""
    _ = dm_sparse_intra_padding if sparse_mode else dm_dense_intra_padding
    return _(input_array, dist_function, condensed=condensed)
