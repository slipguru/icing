# import os
import numpy as np
import scipy
import scipy.spatial
# import time, re, sys
import multiprocessing as mp
# import ctypes as c

from .utils.utils import _terminate, progressbar


def dist2nearest_dual_padding(l1, l2, dist_function):
    """Compute in a parallel way a dist2nearest for two 1-d arrays.
    
    Use this function with different arrays; if l1 == l2, then the
    results is a 0-array.

    Parameters
    ----------
    l1, l2 : array_like
        1-dimensional arrays. Compute the nearest element of l2 to l1.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist2nearest : array_like
        1-D array
    """
    def _internal(l1, l2, n, m, idx, nprocs, shared_arr, dist_function):
        for i in range(idx, n, nprocs):
            if i % 100 == 0: 
                progressbar(i, n)
            shared_arr[i] = min((dist_function(l1[i], l2[j]) for j in range(m)))
    
    n, m = len(l1), len(l2)
    nprocs = min(mp.cpu_count(), n)
    shared_array = mp.Array('d', [0.]*n)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(l1, l2, n, m, idx, shared_array,
                                 dist_function))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
    except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
    except: _terminate(ps,'ERROR: Exiting with unknown exception\n')

    progressbar(n, n)
    return shared_array
    
def dist2nearest_intra_padding(l1, dist_function):
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
            if i % 100 == 0: 
                progressbar(i, n)
            _min1 = min((dist_function(l1[i], l1[j]) for j in range(0, idx)))
            _min2 = min((dist_function(l1[i], l1[j]) for j in range(idx+1, m)))
            shared_arr[i] = min(_min1, _min2)
    
    n = len(l1)
    nprocs = min(mp.cpu_count(), n)
    shared_array = mp.Array('d', [0.]*n)
    ps = []
    try:
        for idx in range(nprocs):
            p = mp.Process(target=_internal,
                           args=(l1, n, idx, shared_array, dist_function))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
    except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
    except: _terminate(ps,'ERROR: Exiting with unknown exception\n')

    progressbar(n, n)
    return shared_array


def _dist2nearest_dual(lock, list1, list2, global_idx, shared_arr, dist_function):
    """Parallelize a general computation of a distance matrix.

    Parameters
    ----------
    lock : multiprocessing.synchronize.Lock
        Value returned from multiprocessing.Lock().
    input_list : list
        List of values to compare to input_list[idx] (from 'idx' on).
    shared_arr : array_like
        Numpy array created as a shared object. Iteratively updated with the result.
        Example:
            shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))

    Returns
    -------

    """
    list_len = len(list1)
    # PID = os.getpid()
    # print("PID {} takes index {}".format(PID, index_i))
    while global_idx.value < list_len:
        with lock:
            if not global_idx.value < list_len: return
            idx = global_idx.value
            global_idx.value += 1
            if idx % 100 == 0: progressbar(idx, list_len)
        elem_1 = list1[idx]
        for idx_j in range(len(list2)):
            _aux = dist_function(elem_1, list2[idx_j])
            if _aux != 0:
                _prev = shared_arr[idx]
                if _prev == 0 or _prev > _aux:
                    shared_arr[idx] = _aux


def dist2nearest_dual(list1, list2, dist_function):
    """Compute in a parallel way a dist2nearest (without 0 vals) for two 1-d
    arrays.

    Parameters
    ----------
    input_array : array_like
        1-dimensional array for which to compute the distance matrix.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist2nearest : array_like
        1-D array
    """
    n = len(list1)
    n_proc = min(mp.cpu_count(), n)
    index = mp.Value('i', 0)
    shared_array = mp.Array('d', [0.]*n)
    ps = []
    lock = mp.Lock()
    try:
        for _ in range(n_proc):
            p = mp.Process(target=_dist2nearest_dual,
                           args=(lock, list1, list2, index, shared_array,
                                 dist_function))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
    except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
    except: _terminate(ps,'ERROR: Exiting with unknown exception\n')

    progressbar(n, n)
    return shared_array


def _dense_distance_dual(lock, list1, list2, global_idx, shared_arr, dist_function):
    """Parallelize a general computation of a distance matrix.

    Parameters
    ----------
    lock : multiprocessing.synchronize.Lock
        Value returned from multiprocessing.Lock().
    input_list : list
        List of values to compare to input_list[idx] (from 'idx' on).
    shared_arr : array_like
        Numpy array created as a shared object. Iteratively updated with the result.
        Example:
            shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))

    Returns
    -------

    """
    list_len = len(list1)
    # PID = os.getpid()
    # print("PID {} takes index {}".format(PID, index_i))
    while global_idx.value < list_len:
        with lock:
            if not global_idx.value < list_len: return
            idx = global_idx.value
            global_idx.value += 1
            #if idx % 100 == 0: progressbar(idx, list_len)
        elem_1 = list1[idx]
        for idx_j in range(len(list2)):
            shared_arr[idx, idx_j] = dist_function(elem_1, list2[idx_j])


def dense_dm_dual(list1, list2, dist_function, condensed=False):
    """Compute in a parallel way a distance matrix for a 1-d array.

    Parameters
    ----------
    input_array : array_like
        1-dimensional array for which to compute the distance matrix.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Symmetric NxN distance matrix for each input_array element.
    """
    n, m = len(list1), len(list2)
    n_proc = min(mp.cpu_count(), n)
    index = mp.Value('i', 0)
    shared_array = np.frombuffer(mp.Array('d', n*m).get_obj()).reshape((n,m))
    ps = []
    lock = mp.Lock()
    try:
        for _ in range(n_proc):
            p = mp.Process(target=_dense_distance_dual,
                        args=(lock, list1, list2, index, shared_array, dist_function))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
    except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
    except: _terminate(ps,'ERROR: Exiting with unknown exception\n')

    dist_matrix = shared_array.flatten() if condensed else shared_array
    #progressbar(n,n)
    return dist_matrix


def _dense_distance(lock, input_list, global_idx, shared_arr, dist_function):
    """Parallelize a general computation of a distance matrix.

    Parameters
    ----------
    lock : multiprocessing.synchronize.Lock
        Value returned from multiprocessing.Lock().
    input_list : list
        List of values to compare to input_list[idx] (from 'idx' on).
    shared_arr : array_like
        Numpy array created as a shared object. Iteratively updated with the result.
        Example:
            shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))

    Returns
    -------

    """
    list_len = len(input_list)
    # PID = os.getpid()
    # print("PID {} takes index {}".format(PID, index_i))
    while global_idx.value < list_len:
        with lock:
            if not global_idx.value < list_len: return
            idx = global_idx.value
            global_idx.value += 1
            if (idx) % 100 == 0: progressbar(idx, list_len)

        elem_1 = input_list[idx]
        for idx_j in range(idx+1, list_len):
            shared_arr[idx, idx_j] = dist_function(elem_1, input_list[idx_j])


def dense_dm(input_array, dist_function, condensed=False):
    """Compute in a parallel way a distance matrix for a 1-d array.

    Parameters
    ----------
    input_array : array_like
        1-dimensional array for which to compute the distance matrix.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Symmetric NxN distance matrix for each input_array element.
    """
    n = len(input_array)
    n_proc = min(mp.cpu_count(), n)
    index = mp.Value('i', 0)
    shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))
    # np.savetxt("shared_array", shared_array, fmt="%.2f", delimiter=',')
    ps = []
    lock = mp.Lock()
    try:
        for _ in range(n_proc):
            p = mp.Process(target=_dense_distance,
                        args=(lock, input_array, index, shared_array, dist_function))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
    except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
    except: _terminate(ps,'ERROR: Exiting with unknown exception\n')

    dist_matrix = shared_array + shared_array.T
    if condensed: dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
    progressbar(n,n)
    return dist_matrix


def _sparse_distance(lock, input_list, global_idx, rows, cols, data, dist_function):
    """Parallelize a general computation of a sparse distance matrix.

    Parameters
    ----------
    lock : multiprocessing.synchronize.Lock
        Value returned from multiprocessing.Lock().
    input_list : list
        List of values to compare to input_list[idx] (from 'idx' on).
    shared_arr : array_like
        Numpy array created as a shared object. Iteratively updated with the result.
        Example:
            shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))

    Returns
    -------

    """
    list_len = len(input_list)
    # PID = os.getpid()
    # print("PID {} takes index {}".format(PID, index_i))
    while global_idx.value < list_len:
        with lock:
            if not global_idx.value < list_len: return
            idx = global_idx.value
            global_idx.value += 1
            if (idx) % 100 == 0: progressbar(idx, list_len)

        elem_1 = input_list[idx]
        for idx_j in range(idx+1, list_len):
             _res = dist_function(elem_1, input_list[idx_j])
             if _res > 0:
                 i, j, d = idx, idx_j, list_len
                 c_idx = d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1
                 data[c_idx] = _res
                 rows[c_idx] = i
                 cols[c_idx] = j



def sparse_dm(input_array, dist_function, condensed=False):
    """Compute in a parallel way a distance matrix for a 1-d input array.

    Parameters
    ----------
    input_array : array_like
        1-dimensional array for which to compute the distance matrix.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Sparse symmetric NxN distance matrix for each input_array element.
    """
    n = len(input_array)
    nproc = min(mp.cpu_count(), n)
    idx = mp.Value('i', 0)
    c_length = int(n*(n-1)/2)
    data = mp.Array('d', [0.]*c_length)
    rows = mp.Array('d', [0.]*c_length)
    cols = mp.Array('d', [0.]*c_length)
    ps = []
    lock = mp.Lock()
    try:
        for _ in range(nproc):
            p = mp.Process(target=_sparse_distance,
                        args=(lock, input_array, idx, rows, cols, data,
                              dist_function))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
    except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
    except: _terminate(ps,'ERROR: Exiting with unknown exception\n')

    data = np.array(data)
    idx = data > 0
    data = data[idx]
    rows = np.array(rows)[idx]
    cols = np.array(cols)[idx]
    # print (data)
    D = scipy.sparse.csr_matrix((data,(rows,cols)), shape=(n,n))
    dist_matrix = D + D.T
    if condensed: dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
    progressbar(n,n)
    return dist_matrix


def _sparse_distance_opt(lock, input_list, global_idx, rows, cols, data, func):
    """Parallelize a general computation of a sparse distance matrix.

    Parameters
    ----------
    lock : multiprocessing.synchronize.Lock
        Value returned from multiprocessing.Lock().
    input_list : list
        List of values to compare to input_list[idx] (from 'idx' on).
    shared_arr : array_like
        Numpy array created as a shared object. Iteratively updated with the result.
        Example:
            shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))

    Returns
    -------

    """
    list_len = input_list.shape[0]
    # PID = os.getpid()
    # print("PID {} takes index {}".format(PID, index_i))
    while global_idx.value < list_len:
        with lock:
            if not global_idx.value < list_len: return
            idx = global_idx.value
            global_idx.value += 1
            if (idx) % 100 == 0: progressbar(idx, list_len)

        for i in range(idx, list_len-1):
             _res = func(input_list[i], input_list[i + 1])
             if _res > 0:
                 j, d = i+1, list_len
                 c_idx = d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1
                 data[c_idx] = _res
                 rows[c_idx] = i
                 cols[c_idx] = j



def sparse_dm_opt(input_array, dist_function, condensed=False):
    """Compute in a parallel way a distance matrix for a 1-d input array.

    Parameters
    ----------
    input_array : array_like
        1-dimensional array for which to compute the distance matrix.
    dist_function : function
        Function to use for the distance computation.

    Returns
    -------
    dist_matrix : array_like
        Sparse symmetric NxN distance matrix for each input_array element.
    """
    n = input_array.shape[0]
    nproc = min(mp.cpu_count(), n)
    idx = mp.Value('i', 0)
    c_length = int(n*(n-1)/2)
    data = mp.Array('d', [0.]*c_length)
    rows = mp.Array('d', [0.]*c_length)
    cols = mp.Array('d', [0.]*c_length)
    ps = []
    lock = mp.Lock()
    try:
        for _ in range(nproc):
            p = mp.Process(target=_sparse_distance_opt,
                           args=(lock, input_array, idx, rows, cols, data,
                                 dist_function))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
    except (KeyboardInterrupt, SystemExit): _terminate(ps, 'Exit signal received\n')
    except Exception as e: _terminate(ps, 'ERROR: %s\n' % e)
    except: _terminate(ps, 'ERROR: Exiting with unknown exception\n')

    data = np.array(data)
    idx = data > 0
    data = data[idx]
    rows = np.array(rows)[idx]
    cols = np.array(cols)[idx]
    # print (data)
    D = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    dist_matrix = D + D.T
    if condensed: dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
    progressbar(n, n)
    return dist_matrix


def distance_matrix_parallel(input_array, dist_function, condensed=False, sparse_mode=False):
    _ = sparse_dm if sparse_mode else dense_dm
    return _(input_array, dist_function, condensed=condensed)
