#import numpy as np
cimport numpy as np
cimport openmp
cimport cython

from numpy cimport ndarray
from cython.parallel cimport parallel, prange, threadid
from ctypes import c_double as double
from cpython cimport array
from cpython.string cimport PyString_AsString, PyString_AS_STRING

from libc.stdio cimport printf, fprintf, stderr
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
    int globalxx(const char * a, const char * b, char * a_n, char * b_n) nogil;
    double cdist_function(const char * a, const char * b) nogil;

cdef char ** to_cstring_array(list_str):
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyString_AsString(list_str[i])
    return ret

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
    cdef Py_ssize_t n = len(list3), m = len(list4)
    cdef Py_ssize_t i, j
    cdef double * M = <double *> malloc(sizeof(double) * n * m);
    if not M:
        raise MemoryError()
    cdef char * elem_1

    cdef char ** list1 = <char**> malloc(n * sizeof(char*));
    if not list1:
        raise MemoryError()
    for i in xrange(n):
        list1[i] = PyString_AsString(list3[i])
        # fprintf(stderr, "list1 - before %s -- %x\n", list1[i], list1[i])

    cdef char ** list2 = <char**> malloc(m * sizeof(char*));
    if not list2:
        raise MemoryError()
    for i in xrange(m):
        # fprintf(stderr, "list1 - meanwhile before %s -- %x\n", list1[i], list1[i])
        list2[i] = PyString_AsString(list4[i])
        # fprintf(stderr, "list2 - before %s -- %x\n", list2[i], list2[i])
        # fprintf(stderr, "list1 - meanwhile after %s -- %x\n", list1[i], list1[i])

    # for i in range(n*m):
        # M[i] = 0.0
    # cdef int num_threads
    # openmp.private(elem_1)
    # openmp.omp_set_dynamic(1)
    for i in prange(n, nogil=True):
        #print("I am %i of %i" % (threadid(), openmp.omp_get_num_threads()))
        elem_1 = list1[i]
        for j in range(m):
            fprintf(stderr, "Analising %s and %s\n", elem_1, list2[j])
            M[i*m+j] = cdist_function(elem_1, list2[j])

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
