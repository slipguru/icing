/* author: Federico Tomasi
 * license: FreeBSD License
 * copyright: Copyright (C) 2016 Federico Tomasi
 */
#include <Python.h>
#include "alignment.h"
int i, j, k;

static void strrev(char *p) {
  char *q = p;
  while(q && *q) ++q; /* find eos */
  for(--q; p < q; ++p, --q) SWP(*p, *q);
}

static void print_alignment_matrix(int * M, int n, int m, const char * a, const char * b) {
    for(j = 0, printf("  "); j < m; ++j) {
    	printf("   %c", b[j]);
    }
    printf("\n");
    for(i = 0; i < n; ++i, printf("\n")) {
        printf(" %c", a[i]);
    	for(j = 0; j < m; ++j) {
    		printf(" %3i", M[i*m + j]);
    	}
    }
}

static int match_fn(const char a, const char b) {
    return (a == b) ? MATCH : MISMATCH;
}

static int globalxx(const char * const a, const char * const b, char * a_n, char * b_n) {
    /*Alignment using Needleman-Wunsch algorithm. */
    int len_a = strlen(a);
    int len_b = strlen(b);
    int n = len_a;
    int m = len_b;
    int row, col;

	//Create empty table
    int * matrix = (int *)malloc(n*m*sizeof(int));
	for(j = 0; j < m; ++j) { // initialise first row
		matrix[j] = match_fn(a[0], b[j]);
	}
	for(i = 0; i < n; ++i) { // initialise first col
		matrix[i*m] = match_fn(a[i], b[0]);
	}
    int best_score, best_i, best_j;
    int score;
	for(row = 1; row < n; ++row) {
		for(col = 1; col < m; ++col) {
            best_score = matrix[(row-1)*m + col-1];
            best_i = row-1;
            best_j = col-1;

            for(i = 0; i < col-1; ++i) {
                score = matrix[(row-1)*m + i];
                if(score > best_score) {
                    best_score = score;
                    best_i = row-1;
                    best_j = i;
                }
            }
            for(i = 0; i < row-1; ++i) {
                score = matrix[i*m+col-1];
                if(score > best_score) {
                    best_score = score;
                    best_i = i;
                    best_j = col-1;
                }
            }
			matrix[row*m + col] = best_score + match_fn(a[row], b[col]);
		}
	}

    // Find global start
    // 1. Search all rows in the last column.
    best_score = matrix[m-1]; // initialise with last element TODO
    best_i = n-1;
    best_j = m-1;
    for(row = 0; row < n; ++row) {
        score = matrix[(row)*m + m-1];
        if(score > best_score) {
            best_score = score;
            best_i = row;
            best_j = m-1;
        }
    }
    // 2. Search all columns in the last row.
    for(col = 0; col < m; ++col) {
        score = matrix[(n-1)*m + col];
        if(score > best_score) {
            best_score = score;
            best_i = n-1;
            best_j = col;
        }
    }

    int cont = 0; // tiene l'ultimo carattere copiato in a_n e b_n
                  // (devono avere sempre la stessa lunghezza)

    // Set first gaps
    int nseqA, nseqB, maxseq, ngapA, ngapB;
    if (best_i != n-1 || best_j != m-1) {
        nseqA = n-1-best_i;
        nseqB = m-1-best_j;
        maxseq = nseqA > nseqB ? nseqA : nseqB;
        ngapA = maxseq - nseqA;
        ngapB = maxseq - nseqB;
        for(k=0; k < ngapA; ++k) a_n[cont+k] = GAP;
        for(k=0; k < nseqA; ++k) a_n[cont+k] = a[n-1-k];
        for(k=0; k < ngapB; ++k) b_n[cont+k] = GAP;
        for(k=0; k < nseqB; ++k) b_n[cont+k] = b[m-1-k];
        cont = maxseq;
    }

    a_n[cont] = a[best_i];
    a_n[cont+1] = '\0';
    b_n[cont] = b[best_j];
    b_n[cont+1] = '\0';
    int relmax, relmax_i, relmax_j;
	while(best_i > 0 && best_j > 0) {
        cont++;
    	relmax = matrix[(best_i-1)*m + best_j-1];
        relmax_i = best_i-1;
        relmax_j = best_j-1;
        if(relmax < matrix[(best_i-1)*m + best_j]) {
          relmax = matrix[(best_i-1)*m + best_j];
          relmax_i = best_i-1;
          relmax_j = best_j;
        }
        if(relmax < matrix[(best_i)*m + best_j-1]) {
          relmax = matrix[(best_i)*m + best_j-1];
          relmax_i = best_i;
          relmax_j = best_j-1;
        }

    	if((relmax_i == best_i-1) && (relmax_j == best_j-1)) {
            //if relmax position is diagonal from current position simply align
            a_n[cont] = a[relmax_i];
    		b_n[cont] = b[relmax_j];
    	} else {
            if(relmax_j == best_j-1) {
                // value on the left, a remains fixed and a_n needs a GAP
                a_n[cont] = GAP;
        		b_n[cont] = b[relmax_j];
            } else if(relmax_i == best_i-1) {
                // value on the top, b remains fixed and a_n needs a GAP
                a_n[cont] = a[relmax_i];
        		b_n[cont] = GAP;
            }
    	}
        a_n[cont+1] = '\0';
        b_n[cont+1] = '\0';
        // fprintf(stderr, "a_n: %s, b_n: %s\n", a_n, b_n);
        best_i = relmax_i;
        best_j = relmax_j;
    }
    // print_alignment_matrix(matrix, n,  m, a, b);
    // fprintf(stderr, "PRIMA1!  %s and %s, %i, %i\n", a_n, b_n, best_i, best_j);
    while(best_i > 0) {
        // complete with the remaining chars
        cont++;
        best_i--;
        a_n[cont] = a[best_i];
        b_n[cont] = GAP;
    }
    while(best_j > 0) { // one or the other
        // complete with the remaining chars
        cont++;
        best_j--;
        b_n[cont] = b[best_j];
        a_n[cont] = GAP;
    }
    // fprintf(stderr, "PRIMA3!  %s and %s, %i, %i\n", a_n, b_n, best_i, best_j);
    a_n[cont+1] = '\0';
    b_n[cont+1] = '\0';
    size_t len_a_n = strlen(a_n), len_b_n = strlen(b_n);
    if(len_a_n < len_b_n) {
        // assert they have the same length, pad a_n with GAP chars
        for(k = len_a_n; k < len_b_n; ++k) {
            a_n[k] = GAP;
        }
        a_n[len_b_n] = '\0';
    } else if (len_a_n > len_b_n) {
        // assert they have the same length, pad b_n with GAP chars
        for(k = len_b_n; k < len_a_n; ++k) {
            b_n[k] = GAP;
        }
        b_n[len_a_n] = '\0';
    }
    free(matrix);
    strrev(a_n);
    strrev(b_n);
	return 0;
}

static PyObject * alignment(PyObject *self, PyObject *args) {
    const char *s1;
    const char *s2;
    char *s1_new;
    char *s2_new;

    if (!PyArg_ParseTuple(args, "ss", &s1, &s2))
        return NULL;

    size_t len_a = strlen(s1), len_b = strlen(s2);
    s1_new = (char *)malloc(sizeof(char)*(len_a+len_b+1));
    s2_new = (char *)malloc(sizeof(char)*(len_a+len_b+1));
    if(!s1_new || !s2_new) {
        fprintf(stderr, "Error, malloc failed");
    }
    s1_new[0] = '\0';
    s2_new[0] = '\0';
    globalxx(s1,s2,s1_new,s2_new);
    // fprintf(stderr, "DOPO! %s and %s\n", s1_new, s2_new);
    PyObject * ret = Py_BuildValue("ss", s1_new, s2_new);
    free(s1_new);
    free(s2_new);
    return ret;
}

static PyMethodDef AlignMethods[] = {
    {"alignment",  alignment, METH_VARARGS,
     "docs."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initalign(void) {
    (void) Py_InitModule("align", AlignMethods);
}

int main(int argc, char *argv[]) {
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initalign();
    return 0;
}
