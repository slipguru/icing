cimport numpy as np
np.import_array() # initialize C API to call PyArray_SimpleNewFromData

cdef public api tonumpyarray(double* data, long long size) with gil:
    if not (data and size >= 0): raise ValueError
    cdef np.npy_intp dims = size
    #NOTE: it doesn't take ownership of `data`. You must free `data` yourself
    return np.PyArray_SimpleNewFromData(1, &dims, np.NPY_DOUBLE, <void*>data)
