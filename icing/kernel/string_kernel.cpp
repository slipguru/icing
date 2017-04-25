/* author: Federico Tomasi
 * license: FreeBSD License
 * copyright: Copyright (C) 2016 Federico Tomasi
 */
#include <Python.h>
#include "file_io.h"
#include "string_kernel.h"
#include <sstream>

using std::cerr;
using std::endl;
using std::cout;
using std::string;
using std::vector;

static PyObject * string_kernel(PyObject *self, PyObject *args, PyObject *keywds) {
    // const char *s1;
    // const char *s2;
    // const char *s3;
    // // double result;
    // if (!PyArg_ParseTuple(args, "sss", &s1, &s2, &s3))
    //     return NULL;

    // string kernel_file(s1);

    // Kernel parameters
    const float c = 1e12;  // unused
    int normalize = 1;
    const int symbol_size = 255;  // A size of an alphabet
    const int max_length = 1000;  // A maximum sequence length
    size_t kn = 2;                   // A level of susbsequence matching
    double lambda = .5;          // A decay factor

    // Prepare dummy data
    vector<string> dummy_data;

    // Prepare labels for dummy data
    vector<string> dummy_labels;

    int numLines;       /* how many lines we passed for parsing */
    char * line;        /* pointer to the line as a string */



    PyObject * listObj; /* the list of strings */
    PyObject * strObj;  /* one string in the list */
    PyObject * labels = NULL; /* the list of strings */
    char * filename = (char *)"output.txt"; // default value

    static char *kwlist[] = {(char*)"sequences", (char*)"filename", (char*)"normalize",
                             (char*)"kn", (char*)"lamda", (char*)"labels", NULL};
    /* the O! parses for a Python object (listObj) checked to be of type PyList_Type */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|siidO!", kwlist,
                                     &PyList_Type, &listObj, &filename,
                                     &normalize, &kn, &lambda,
                                     &PyList_Type, &labels))
        return NULL;

    /* get the number of lines passed to us */
    numLines = PyList_Size(listObj);
    if (numLines < 0) return NULL; /* Not a list */

    string kernel_file(filename);
    string label;
    std::stringstream ss;
    for (int i = 0; i < numLines; i++){
    	/* grab the string object from the next element of the list */
    	strObj = PyList_GetItem(listObj, i); /* Can't fail */
    	line = PyString_AsString( strObj );
        dummy_data.push_back(line);

        if (labels != NULL) {
            strObj = PyList_GetItem(labels, i);
        	line = PyString_AsString(strObj);
            label = string(line);
        } else {
            ss << i;
            label = ss.str();
        }
        dummy_labels.push_back(label);
	}

    std::cout << "Parameters:";
    std::cout << "\n\tfilename: " << filename;
    std::cout << "\n\tnormalize: " << normalize;
    std::cout << "\n\tkn: " << kn;
    std::cout << "\n\tlambda: " << lambda;
    std::cout << "\n";


    // C version
//     char * tok;         /* delimiter tokens for strtok */
//     int cols;           /* number of cols to parse, from the left */
//
//     int numLines;       /* how many lines we passed for parsing */
//     char * line;        /* pointer to the line as a string */
//     char * token;       /* token parsed by strtok */
//
//     PyObject * listObj; /* the list of strings */
//     PyObject * strObj;  /* one string in the list */
//
//     /* the O! parses for a Python object (listObj) checked
//        to be of type PyList_Type */
//     if (! PyArg_ParseTuple( args, "O!is", &PyList_Type, &listObj,
// 			   &cols, &tok )) return NULL;
//
//     /* get the number of lines passed to us */
//     numLines = PyList_Size(listObj);
//
//     /* should raise an error here. */
//     if (numLines < 0)	return NULL; /* Not a list */
//
// ...
//
//     /* iterate over items of the list, grabbing strings, and parsing
//        for numbers */
//     for (i=0; i<numLines; i++){
//
// 	/* grab the string object from the next element of the list */
// 	strObj = PyList_GetItem(listObj, i); /* Can't fail */
//
// 	/* make it a string */
// 	line = PyString_AsString( strObj );
//

    // Main computations
    StringKernel<float> string_kernel(c, normalize, symbol_size, max_length, kn, lambda);
    string_kernel.set_data(dummy_data);
    string_kernel.compute_kernel();

    if (write_kernel(kernel_file, dummy_labels, string_kernel))
      std::cout << "Kernel saved in the libsvm format to: " << kernel_file << std::endl;
    else {
      std::cout << "Error: Cannot write to file: " << kernel_file << std::endl;
      exit(1);
    }

    PyObject * ret = Py_BuildValue("s", "OK");
    return ret;
}

static PyMethodDef StringKernelMethods[] = {
    {"string_kernel", (PyCFunction)string_kernel, METH_VARARGS | METH_KEYWORDS,
     "docs."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initstring_kernel(void) {
    (void) Py_InitModule("string_kernel", StringKernelMethods);
}

int main(int argc, char *argv[]) {
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initstring_kernel();
    return 0;
}
