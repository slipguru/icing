/* author: Federico Tomasi
 * license: FreeBSD License
 * copyright: Copyright (C) 2016 Federico Tomasi
 */
#include <Python.h>
#include "file_io.h"
#include "string_kernel.h"
#include "sum_string_kernel.h"
#include <sstream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// static PyArrayObject *
static PyObject *
stringkernel(PyObject *self, PyObject *args, PyObject *keywds) {
    // Kernel parameters
    int normalize = 1;
    int verbose = 0;
    int save_output = 0;
    int return_float = 0;
    int check_min_length = 0;
    int hard_matching = 0;
    const int symbol_size = 255;  // A size of an alphabet
    const int max_length = 1000;  // A maximum sequence length
    size_t min_kn = 1;                   // A level of subsequence matching
    size_t max_kn = 2;                   // A level of subsequence matching
    double lambda = .5;          // A decay factor

    // Prepare data
    std::vector<std::string> vector_data;
    std::vector<std::string> vector_labels;

    Py_ssize_t list_size;      /* how many lines we passed for parsing */
    char * line;        /* pointer to the line as a string */

    PyObject * listObj; /* the list of strings */
    PyObject * strObj;  /* one string in the list */
    PyObject * labels = NULL; /* the list of strings */
    char * filename = (char *)"output.txt"; // default value

    static char *kwlist[] = {
        (char*)"sequences", (char*)"filename", (char*)"normalize",
        (char*)"min_kn", (char*)"max_kn", (char*)"lamda",
        (char*)"save_output", (char*)"hard_matching",
        (char*)"verbose", (char*)"return_float", (char*)"check_min_length",
        (char*)"labels", NULL
    };
    /* the O! parses for a Python object (listObj) checked to be of type PyList_Type */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|siiidiiiiiO!", kwlist,
            &PyList_Type, &listObj, &filename, &normalize, &min_kn, &max_kn,
            &lambda, &save_output, &hard_matching, &verbose, &return_float,
            &check_min_length,
            &PyList_Type, &labels))
        return NULL;

    /* get the number of lines passed */
    list_size = PyList_Size(listObj);
    if (list_size < 0) return NULL; /* Not a list */

    std::string kernel_file(filename);
    std::string label;
    std::stringstream ss;
    for (Py_ssize_t i = 0; i < list_size; i++){
    	/* grab the string object from the next element of the list */
    	strObj = PyList_GetItem(listObj, i); /* Can't fail */
    	line = PyString_AsString(strObj);  /* make it a string */
        if(check_min_length && strlen(line) < min_kn) {
            min_kn = strlen(line);
        }
        vector_data.push_back(line);

        if (labels != NULL) {
            strObj = PyList_GetItem(labels, i);
        	line = PyString_AsString(strObj);
            label = std::string(line);
        } else {
            // convert i into a string
            ss << i;
            label = ss.str();
        }
        vector_labels.push_back(label);
	}

    if(verbose) {
        std::cout << "Parameters:"
        << "\n\tfilename: " << filename
        << "\n\tnormalize: " << normalize
        << "\n\tmin_kn: " << min_kn
        << "\n\tmax_kn: " << max_kn
        << "\n\tlambda: " << lambda
        << "\n\thard_matching: " << hard_matching
        << std::endl;
    }

    if(min_kn > max_kn) {
        PyErr_SetString(PyExc_ValueError, "`min_kn` is higher than `max_kn`");
        return NULL;
    }

    // Main computations
    SumStringKernel<float> string_kernel(min_kn, max_kn, normalize,
                                         symbol_size, max_length, lambda,
                                         hard_matching);
    string_kernel.set_data(vector_data);
    string_kernel.compute_kernel();

    if (save_output) {
      if (!write_kernel(kernel_file, vector_labels, string_kernel)) {
        PyErr_SetString(PyExc_IOError, "Cannot write to filename specified");
        return NULL;
      }
    }

    if(return_float && list_size == 2) {
        return Py_BuildValue("f", string_kernel.values()[1]);
    }

    // build the numpy matrix to pass back to the Python code
    // need to copy data to prevent the deleting in string kernel class
    // TODO avoid copying data?
    float * data = (float*) malloc(list_size*list_size*sizeof(float));
    if (!data) {
        PyErr_SetString(PyExc_MemoryError, "out of memory");
        return NULL;
    }
    string_kernel.copy_kernel(data);

    npy_intp* size = (npy_intp*)malloc(sizeof(npy_intp)*2);
    size[0] = size[1] = list_size;
    PyObject * py_arr = PyArray_SimpleNewFromData(2, size, NPY_FLOAT, data);
    return py_arr;
}

static PyMethodDef StringKernelMethods[] = {
    {"stringkernel", (PyCFunction)stringkernel, METH_VARARGS | METH_KEYWORDS,
     "docs."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initstringkernel(void) {
    (void) Py_InitModule("stringkernel", StringKernelMethods);
    import_array();  // required to use PyArray_SimpleNewFromData
}

int main(int argc, char *argv[]) {
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initstringkernel();
    return 0;
}
