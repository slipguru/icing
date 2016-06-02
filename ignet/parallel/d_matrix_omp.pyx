#import numpy as np
cimport numpy as np
cimport openmp
cimport cython

from numpy cimport ndarray
from cython.parallel cimport parallel, prange, threadid
from ctypes import c_double as double
from cpython cimport array
from cpython.string cimport PyString_AsString

from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
# from libcpp.string cimport string
# from libcpp.vector cimport vector

np.import_array()
cdef inline int d_min(int a, int b): return a if a <= b else b
ctypedef double (*f_type_dist)(char*, char*) ## to mod

#np.ndarray[bytes, ndim=1] list1,
# cdef extern from "alignment_cpp.h":
#     int local_alignment(string, string, string &, string &) nogil;
#     int global_alignment(string, string, string &, string &) nogil;

cdef extern from "alignment.h":
    int local_alignment(const char * a, const char * b, char * a_n, char * b_n) nogil;
    int global_alignment(const char * a, const char * b, char * a_n, char * b_n) nogil;
    double cdist_function(const char * a, const char * b) nogil;

# cdef char ** to_cstring_array(list_str):
#     cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
#     for i in xrange(len(list_str)):
#         ret[i] = PyString_AsString(list_str[i])
#     return ret

# def foo(list_str1, list_str2):
#     cdef unsigned int i, j
#     cdef char **c_arr1 = to_cstring_array(list_str1)
#     cdef char **c_arr2 = to_cstring_array(list_str2)
#
#     for i in xrange(len(list_str1)):
#         for j in xrange(len(list_str2)):
#             if i != j and strcmp(c_arr1[i], c_arr2[j]) == 0:
#                 print i, j, list_str1[i]
#     free(c_arr1)
#     free(c_arr2)
#
# foo(['hello', 'python', 'world'], ['python', 'rules'])
# cdef string* to_string_array(list_str):
#     cdef string *ret = <string *>malloc(len(list_str) * sizeof(string))
#     for i in xrange(len(list_str)):
#         ret[i] = <string>(list_str[i])
#     return ret

# cdef vector[string] to_string_vector(list_str):
#     cdef vector[string] ret = list_str
#     return ret


# from libc.stdio cimport printf
# from libc.string cimport *
# cdef double cdist_function(const char * a, const char * b) nogil:
#     cdef:
#         size_t len_a = strlen(a), len_b = strlen(b);
#         char * c = <char *>malloc(sizeof(char)*(len_a+len_b+1));
#         char * d = <char *>malloc(sizeof(char)*(len_a+len_b+1));
#     if(c == NULL or d == NULL):
#         printf("Error, malloc failed");
#     global_alignment(a, b, c, d)
#     printf("EHI! %s and %s\n", c, d)
#     free(c)
#     free(d)
#     return <double>(max(strlen(a), strlen(b)))
# cdef double cdist_function(string a, string b) nogil:
#     cdef:
#         size_t len_a = a.size(), len_b = b.size();
#         string c, d;
#
#     global_alignment(a, b, c, d)
#     # printf("%s\n%s", <const char *>c, <const char*>d)
#     return <double>(max((a).size(), b.size()))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double* dist_matrix(list3, list4):
# def dist_matrix(char** list1, Py_ssize_t n,
#                 char** list2, Py_ssize_t m,
#                 dist_function):

    cdef:
        Py_ssize_t n = list3.shape[0], m = list4.shape[0]
        Py_ssize_t i, j
        double * M = <double *> malloc(sizeof(double) * n * m);
    if not M:
        raise MemoryError()

    cdef char ** list1 = <char**> malloc(n * sizeof(char*));
    for i in xrange(n):
        list1[i] = PyString_AsString(list3[i])
    cdef char ** list2 = <char**> malloc((len(list4)) * sizeof(char*));
    for i in xrange(len(list4)):
        list2[i] = PyString_AsString(list4[i])

    # cdef vector[string] list1 = list3
    # cdef vector[string] list2 = list4
    # cdef string * list1 = to_string_array(list3)
    # cdef string * list2 = to_string_array(list4)

    for i in xrange(n*m):
        M[i] = 0.0

    cdef int num_threads
    cdef char * elem_1

    #openmp.omp_set_dynamic(1)
    for i in prange(n, nogil=True):
        elem_1 = list1[i]
        #print("I am %i of %i" % (threadid(), openmp.omp_get_num_threads()))

        #with gil:
        #    M[i*m:(i+1)*m] = array.array('d', d_f(elem_1, list4, dist_function)).data.as_doubles
        for j in range(m):
            #pass
            # M[i*m+j] = cdist_function(elem_1.c_str(), list2[j].c_str())
            M[i*m+j] = cdist_function(elem_1, list2[j])


    #for i in xrange(len(list3)):
    #    free(list1[i])

    #for i in xrange(len(list4)):
    #    free(list2[i])
    free(list1)
    free(list2)
    return M

cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp N, int t):
    cdef np.ndarray[double, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &N, t, ptr)
    return arr

def wrapper(list3, list4):
    a = dist_matrix(list3, list4)
    arr = data_to_numpy_array_with_spec(a, len(list3)*len(list4), np.NPY_DOUBLE)
    return arr

def d_f(el, list, dist_function):
    return map(lambda x: dist_function(el,x), list)



# def wrapper(func):
#     cdef char *inp
#     cdef np.ndarray a
#     a = np.array(['aaaa', 'bbbbbb'])
#     inp = a.data
#     print a
#     #dist_matrix(inp, a.itemsize*a.shape[0],inp, a.itemsize*a.shape[0],func)
#     dist_matrix(inp, a.itemsize*a.shape[0],inp, a.itemsize*a.shape[0],func)
#     print a
    #return dist_matrix(a1.data, a1.itemsize*a1.shape[0],
    #                   a1.data, a1.itemsize*a1.shape[0],
    #                   func)
##########


# def _dense_distance_dual(lock, list1, list2, global_idx, shared_arr, dist_function):
#     """Parallelize a general computation of a distance matrix.
#
#     Parameters
#     ----------
#     lock : multiprocessing.synchronize.Lock
#         Value returned from multiprocessing.Lock().
#     input_list : list
#         List of values to compare to input_list[idx] (from 'idx' on).
#     shared_arr : array_like
#         Numpy array created as a shared object. Iteratively updated with the result.
#         Example:
#             shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))
#
#     Returns
#     -------
#
#     """
#     list_len = len(list1)
#     # PID = os.getpid()
#     # print("PID {} takes index {}".format(PID, index_i))
#     while global_idx.value < list_len:
#         with lock:
#             if not global_idx.value < list_len: return
#             idx = global_idx.value
#             global_idx.value += 1
#             if idx % 100 == 0: progressbar(idx, list_len)
#         elem_1 = list1[idx]
#         for idx_j in range(len(list2)):
#             shared_arr[idx, idx_j] = dist_function(elem_1, list2[idx_j])
#
#
# def dense_dm_dual(list1, list2, dist_function, condensed=False):
#     """Compute in a parallel way a distance matrix for a 1-d array.
#
#     Parameters
#     ----------
#     input_array : array_like
#         1-dimensional array for which to compute the distance matrix.
#     dist_function : function
#         Function to use for the distance computation.
#
#     Returns
#     -------
#     dist_matrix : array_like
#         Symmetric NxN distance matrix for each input_array element.
#     """
#     n, m = len(list1), len(list2)
#     n_proc = min(mp.cpu_count(), n)
#     index = mp.Value('i', 0)
#     shared_array = np.frombuffer(mp.Array('d', n*m).get_obj()).reshape((n,m))
#     ps = []
#     lock = mp.Lock()
#     try:
#         for _ in range(n_proc):
#             p = mp.Process(target=_dense_distance_dual,
#                         args=(lock, list1, list2, index, shared_array, dist_function))
#             p.start()
#             ps.append(p)
#
#         for p in ps:
#             p.join()
#     except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
#     except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
#     except: _terminate(ps,'ERROR: Exiting with unknown exception\n')
#
#     dist_matrix = shared_array.flatten() if condensed else shared_array
#     progressbar(n,n)
#     return dist_matrix
#
#
# def _dense_distance(lock, input_list, global_idx, shared_arr, dist_function):
#     """Parallelize a general computation of a distance matrix.
#
#     Parameters
#     ----------
#     lock : multiprocessing.synchronize.Lock
#         Value returned from multiprocessing.Lock().
#     input_list : list
#         List of values to compare to input_list[idx] (from 'idx' on).
#     shared_arr : array_like
#         Numpy array created as a shared object. Iteratively updated with the result.
#         Example:
#             shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))
#
#     Returns
#     -------
#
#     """
#     list_len = len(input_list)
#     # PID = os.getpid()
#     # print("PID {} takes index {}".format(PID, index_i))
#     while global_idx.value < list_len:
#         with lock:
#             if not global_idx.value < list_len: return
#             idx = global_idx.value
#             global_idx.value += 1
#             if (idx) % 100 == 0: progressbar(idx, list_len)
#
#         elem_1 = input_list[idx]
#         for idx_j in range(idx+1, list_len):
#             shared_arr[idx, idx_j] = dist_function(elem_1, input_list[idx_j])
#
#
# def dense_dm(input_array, dist_function, condensed=False):
#     """Compute in a parallel way a distance matrix for a 1-d array.
#
#     Parameters
#     ----------
#     input_array : array_like
#         1-dimensional array for which to compute the distance matrix.
#     dist_function : function
#         Function to use for the distance computation.
#
#     Returns
#     -------
#     dist_matrix : array_like
#         Symmetric NxN distance matrix for each input_array element.
#     """
#     n = len(input_array)
#     n_proc = min(mp.cpu_count(), n)
#     index = mp.Value('i', 0)
#     shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))
#     # np.savetxt("shared_array", shared_array, fmt="%.2f", delimiter=',')
#     ps = []
#     lock = mp.Lock()
#     try:
#         for _ in range(n_proc):
#             p = mp.Process(target=_dense_distance,
#                         args=(lock, input_array, index, shared_array, dist_function))
#             p.start()
#             ps.append(p)
#
#         for p in ps:
#             p.join()
#     except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
#     except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
#     except: _terminate(ps,'ERROR: Exiting with unknown exception\n')
#
#     dist_matrix = shared_array + shared_array.T
#     if condensed: dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
#     progressbar(n,n)
#     return dist_matrix
#
#
# def _sparse_distance(lock, input_list, global_idx, rows, cols, data, dist_function):
#     """Parallelize a general computation of a sparse distance matrix.
#
#     Parameters
#     ----------
#     lock : multiprocessing.synchronize.Lock
#         Value returned from multiprocessing.Lock().
#     input_list : list
#         List of values to compare to input_list[idx] (from 'idx' on).
#     shared_arr : array_like
#         Numpy array created as a shared object. Iteratively updated with the result.
#         Example:
#             shared_array = np.frombuffer(mp.Array('d', n*n).get_obj()).reshape((n,n))
#
#     Returns
#     -------
#
#     """
#     list_len = len(input_list)
#     # PID = os.getpid()
#     # print("PID {} takes index {}".format(PID, index_i))
#     while global_idx.value < list_len:
#         with lock:
#             if not global_idx.value < list_len: return
#             idx = global_idx.value
#             global_idx.value += 1
#             if (idx) % 100 == 0: progressbar(idx, list_len)
#
#         elem_1 = input_list[idx]
#         for idx_j in range(idx+1, list_len):
#              _res = dist_function(elem_1, input_list[idx_j])
#              if _res > 0:
#                  i, j, d = idx, idx_j, list_len
#                  c_idx = d*(d-1)/2 - (d-i)*(d-i-1)/2 + j - i - 1
#                  data[c_idx] = _res
#                  rows[c_idx] = i
#                  cols[c_idx] = j
#
#
#
# def sparse_dm(input_array, dist_function, condensed=False):
#     """Compute in a parallel way a distance matrix for a 1-d input array.
#
#     Parameters
#     ----------
#     input_array : array_like
#         1-dimensional array for which to compute the distance matrix.
#     dist_function : function
#         Function to use for the distance computation.
#
#     Returns
#     -------
#     dist_matrix : array_like
#         Sparse symmetric NxN distance matrix for each input_array element.
#     """
#     n = len(input_array)
#     nproc = min(mp.cpu_count(), n)
#     idx = mp.Value('i', 0)
#     c_length = int(n*(n-1)/2)
#     data = mp.Array('d', [0.]*c_length)
#     rows = mp.Array('d', [0.]*c_length)
#     cols = mp.Array('d', [0.]*c_length)
#     ps = []
#     lock = mp.Lock()
#     try:
#         for _ in range(nproc):
#             p = mp.Process(target=_sparse_distance,
#                         args=(lock, input_array, idx, rows, cols, data,
#                               dist_function))
#             p.start()
#             ps.append(p)
#         for p in ps:
#             p.join()
#     except (KeyboardInterrupt, SystemExit): _terminate(ps,'Exit signal received\n')
#     except Exception as e: _terminate(ps,'ERROR: %s\n' % e)
#     except: _terminate(ps,'ERROR: Exiting with unknown exception\n')
#
#     data = np.array(data)
#     idx = data > 0
#     data = data[idx]
#     rows = np.array(rows)[idx]
#     cols = np.array(cols)[idx]
#     # print (data)
#     D = scipy.sparse.csr_matrix((data,(rows,cols)), shape=(n,n))
#     dist_matrix = D + D.T
#     if condensed: dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
#     progressbar(n,n)
#     return dist_matrix
#
#
# def distance_matrix_parallel(input_array, dist_function, condensed=False, sparse_mode=False):
#     _ = sparse_dm if sparse_mode else dense_dm
#     return _(input_array, dist_function, condensed=condensed)
