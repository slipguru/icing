/* Alignment module  */
#include "alignment.h"

/*Definitions*/
int i, j, k;

void strrev(char *p) {
  char *q = p;
  while(q && *q) ++q; /* find eos */
  for(--q; p < q; ++p, --q) SWP(*p, *q);
}

void reverse(char s[]) {
    int length = strlen(s) ;
    int c, i, j;
    for (i = 0, j = length - 1; i < j; i++, j--) {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}

void reverse_string(char *str) {
    /* skip null */
    if (str == 0) { return; }

    /* skip empty string */
    if (*str == 0) { return; }

    /* get range */
    char *start = str;
    char *end = start + strlen(str) - 1; /* -1 for \0 */
    char temp;

    /* reverse */
    while (end > start) {
    	/* swap */
    	temp = *start;
    	*start = *end;
    	*end = temp;

    	/* move */
    	++start;
    	--end;
    }
}

char* stradd(const char* a, const char* b){
    size_t len = strlen(a) + strlen(b);
    char *ret = (char*)malloc(len * sizeof(char) + 1);
    *ret = '\0';
    return strcat(strcat(ret, a) ,b);
}

void print_alignment_matrix(int * M, int n, int m, const char * a, const char * b) {
    for(j = 0, printf("     0"); j < m-1; ++j) {
    	printf("   %c", b[j]);
    }
    printf("\n");
    for(i = 0, printf(" 0"); i < n; ++i, printf("\n")) {
        if(i > 0) printf(" %c", a[i-1]);
    	for(j = 0; j < m; ++j) {
    		printf(" %3i", M[i*m + j]);
    	}
    }
}


int local_alignment(const char * a, const char * b, char * a_n, char * b_n) {
    /*Alignment using Smith-Waterman algorithm*/
    int len_a = strlen(a);
    int len_b = strlen(b);
    int n = len_a + 1;
    int m = len_b + 1;

	//Create empty table
    int * matrix = (int *)malloc(n*m*sizeof(int));
	for(j = 0; j < m; ++j) {
		matrix[j] = 0;
	}
	for(i = 0; i < n; ++i) {
		matrix[i*m] = 0;
	}

	int compval;
	for(i = 1; i < n; ++i) {	//for all values of strA
		for(j = 1; j < m; ++j) {	//for all values of strB
            compval = 0;
			if(a[i-1] == b[j-1]) { //match
				compval = (matrix[(i-1)*m + j-1] + Match);	//compval = diagonal + match score
			} else { //missmatch
                compval = (matrix[(i-1)*m + j-1] + MissMatch);
            }
			for(k = i-1; k > 0; --k) {		//check all sub rows
				if(compval < ((matrix[(i-1)*m + j]) - (GapPenalty + (GapExt * k)))) {
					compval = ((matrix[(i-1)*m + j]) - (GapPenalty + (GapExt * k)));
				}
			}
			for(k = j-1; k > 0; --k) {		//check all sub columns
				if(compval < ((matrix[i*m + j-1]) - (GapPenalty + (GapExt * k)))) {
					compval = ((matrix[i*m + j-1]) - (GapPenalty + (GapExt * k)));
				}
			}
			if(compval < 0) {
				compval = 0;
			}
			matrix[i*m +j] = compval;	//set current cell to highest possible score
		}
	}

	print_alignment_matrix(matrix, n, m, a, b);
	/*MAKE ALIGNMENT*/
    int max_score = matrix[i*m + j];
    int imax = 0, jmax = 0;
	for(i = 0; i < n; ++i) {	//find highest score in matrix: this is the starting point of an optimal local alignment
		for(j = 0; j < m; ++j) {
			if(matrix[i*m + j] > max_score) {
				max_score = matrix[i*m + j];
				imax = i;
				jmax = j;
			}
		}
	}

    a_n[1] = '\0';
    a_n[0] = a[imax-1];
    b_n[1] = '\0';
    b_n[0] = b[jmax-1];
    int relmax = -1;
    int relmaxpos[2];
    int cont = 0;
	while(imax > -1 && jmax > -1) {
        cont++;
    	relmax = matrix[(imax-1)*m + jmax-1];
        relmaxpos[0] = imax-1;
        relmaxpos[1] = jmax-1;
        if(relmax < matrix[(imax-1)*m + jmax]) {
          relmax = matrix[(imax-1)*m + jmax];
          relmaxpos[0] = imax-1;
          relmaxpos[1] = jmax;
        }
        if(relmax < matrix[(imax)*m + jmax-1]) {
          relmax = matrix[(imax)*m + jmax-1];
          relmaxpos[0] = imax;
          relmaxpos[1] = jmax-1;
        }

    	if((relmaxpos[0] == imax-1) && (relmaxpos[1] == jmax-1)) {
            //if relmax position is diagonal from current position simply align
            a_n[cont] = a[relmaxpos[0] - 1]; a_n[cont+1] = '\0';
    		b_n[cont] = b[relmaxpos[1] - 1]; b_n[cont+1] = '\0';
    	} else {
            if((relmaxpos[1] == jmax-1) && (relmaxpos[0] != imax-1)) {
                //maxB needs at least one '-'
                // value on the left
                a_n[cont] = a[imax - 1]; a_n[cont+1] = '\0';
        		b_n[cont] = GAP; b_n[cont+1] = '\0';
            } else if((relmaxpos[0] == imax-1) && (relmaxpos[1] != jmax-1)) {
                //MaxA needs at least one '-'
                a_n[cont] = GAP; a_n[cont+1] = '\0';
        		b_n[cont] = b[jmax - 1]; b_n[cont+1] = '\0';
    		}
    	}
        imax = relmaxpos[0];
        jmax = relmaxpos[1];
    }
    free(matrix);

    /*in the end reverse Max A and B*/
    char * _tmp = malloc(sizeof(char)*(len_a+len_b+1));
	for(i = strlen(a_n)-1, k = 0, strcpy(_tmp, a_n); i > -1; --i, ++k) {
		_tmp[k] = a_n[i];
	}
    strcpy(a_n, _tmp);
	for(j = strlen(b_n)-1, k = 0, strcpy(_tmp, b_n); j > -1; --j, ++k) {
		_tmp[k] = b_n[j];
	}
    strcpy(b_n, _tmp);
    free(_tmp);
	return 0;
}

int global_alignment(const char * const a, const char * const b, char * a_n, char * b_n) {
    /*Alignment using Needleman-Wunsch algorithm. */
    int len_a = strlen(a);
    int len_b = strlen(b);
    int n = len_a + 1;
    int m = len_b + 1;

	//Create empty table
    int * matrix = (int *)malloc(n*m*sizeof(int));
	matrix[0] = 0;
	for(j = 1; j < m; ++j) { // initialise first row
		matrix[j] = matrix[j-1] - GapPenalty;
	}
	for(i = 1; i < n; ++i) { // initialise first col
		matrix[i*m] = matrix[(i-1)*m] - GapPenalty;
	}

    int compval;
	for(i = 1; i < n; ++i) {	//for all values of strA
		for(j = 1; j < m; ++j) {	//for all values of strB
            compval = 0;
			if(a[i-1] == b[j-1]) { //match
				compval = matrix[(i-1)*m + j-1] + Match;	//compval = diagonal + match score
			} else { //missmatch
                compval = matrix[(i-1)*m + j-1] + MissMatch;
            }
            for(k = i-1; k > 0; --k) {		//check all sub rows
				if(compval < ((matrix[(i-1)*m + j]) - (GapPenalty + (GapExt * k)))) {
					compval = ((matrix[(i-1)*m + j]) - (GapPenalty + (GapExt * k)));
				}
			}
			// if(compval < matrix[(i-1)*m + j] - GapPenalty) {
			// 	compval = matrix[(i-1)*m + j] - GapPenalty;
            // }

			for(k = j-1; k > 0; --k) {		//check all sub columns
				if(compval < ((matrix[i*m + j-1]) - (GapPenalty + (GapExt * k)))) {
					compval = ((matrix[i*m + j-1]) - (GapPenalty + (GapExt * k)));
				}
			}
			// if(compval < matrix[(i)*m + j-1] - GapPenalty) {
			// 	compval = matrix[(i)*m + j-1] - GapPenalty;
            // }
			matrix[i*m + j] = compval;	//set current cell to highest possible score
		}
	}
    print_alignment_matrix(matrix, n, m, a, b);

    // Find global start
    // 1. Search all rows in the last column.
    int best_score = matrix[m-1]; // initialise with last element TODO
    int best_indexi = n-1;
    int best_indexj = m-1;
    int row, col, score;
    for(row = 0; row < n; ++row) {
        score = matrix[(row)*m + m-1];
        if(score > best_score) {
            best_score = score;
            best_indexi = row;
            best_indexj = m-1;
        }
    }
    // 2. Search all columns in the last row.
    for(col = 0; col < m; ++col) {
        score = matrix[(n-1)*m + col];
        if(score > best_score) {
            best_score = score;
            best_indexi = n-1;
            best_indexj = col;
        }
    }

    printf("%i %i\n", best_indexi, best_indexj);
    int imax = best_indexi, jmax = best_indexj;

    // a_n = malloc(sizeof(char)*(len_a+len_b+1)); // need space for '\0'
    // b_n = malloc(sizeof(char)*(len_a+len_b+1)); // need space for '\0'
    // if(a_n == NULL || b_n == NULL) {
	// 	fprintf(stderr, "Error, malloc failed");
	// }
    //
    // strcpy(out_a, a[imax-1]);
    // strcpy(a_n, a+n-2);
    // strcpy(out_b, b[imax-1]);

    a_n[1] = '\0';
    a_n[0] = a[imax-1];
    b_n[1] = '\0';
    b_n[0] = b[jmax-1];
    int relmax, relmax_i, relmax_j;
    int cont = 0;
	while(imax > 0 && jmax > 0) {
        cont++;
    	relmax = matrix[(imax-1)*m + jmax-1];
        relmax_i = imax-1;
        relmax_j = jmax-1;
        if(relmax < matrix[(imax-1)*m + jmax]) {
          relmax = matrix[(imax-1)*m + jmax];
          relmax_i = imax-1;
          relmax_j = jmax;
        }
        if(relmax < matrix[(imax)*m + jmax-1]) {
          relmax = matrix[(imax)*m + jmax-1];
          relmax_i = imax;
          relmax_j = jmax-1;
        }

    	if((relmax_i == imax-1) && (relmax_j == jmax-1)) {
            //if relmax position is diagonal from current position simply align
            a_n[cont] = a[relmax_i - 1]; a_n[cont+1] = '\0';
    		b_n[cont] = b[relmax_j - 1]; b_n[cont+1] = '\0';
    	} else {
            if(relmax_j == jmax-1) {
                // value on the left, a remains fixed and a_n needs a GAP
                a_n[cont] = GAP; a_n[cont+1] = '\0';
        		b_n[cont] = b[relmax_j - 1]; b_n[cont+1] = '\0';
            } else if(relmax_i == imax-1) {
                // value on the top, b remains fixed and a_n needs a GAP
                a_n[cont] = a[relmax_i - 1]; a_n[cont+1] = '\0';
        		b_n[cont] = GAP; b_n[cont+1] = '\0';
            }
    	}
        imax = relmax_i;
        jmax = relmax_j;
    }
    free(matrix);

    reverse_string(a_n);
    reverse_string(b_n);
    fprintf(stderr, "a_n = %s\n", a_n);
    fprintf(stderr, "b_n = %s\n", b_n);
	return 0;
}

int match_fn(const char a, const char b) {
    return (a == b) ? MATCH : MISMATCH;
}

int globalxx(const char * const a, const char * const b, char * a_n, char * b_n) {
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
	while(best_i > -1 && best_j > -1) {
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
    // fprintf(stderr, "PRIMA1!  %s and %s\n", a_n, b_n);
    while(best_i > -1) {
        // complete with the remaining chars
        a_n[cont] = a[best_i];
        b_n[cont] = GAP;
        cont++; best_i--;
    }
    while(best_j > -1) {
        // complete with the remaining chars
        b_n[cont] = b[best_j];
        a_n[cont] = GAP;
        cont++; best_j--;
    }
    a_n[cont] = '\0';
    b_n[cont] = '\0';
    int len_a_n = strlen(a_n), len_b_n = strlen(b_n);
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

void align(const char* const a, const char* const b) {
    size_t len_a = strlen(a), len_b = strlen(b);
    char * c = (char *)malloc(sizeof(char)*(len_a+len_b+1));
    char * d = (char *)malloc(sizeof(char)*(len_a+len_b+1));
    if(!c || !d) {
        fprintf(stderr, "Error, malloc failed");
    }
    c[0] = '\0';
    d[0] = '\0';
    globalxx(a, b, c, d);
    fprintf(stderr, "DOPO! %s and %s\n", c, d);
    free(c);
    free(d);
}

double cdist_function(const char* const a, const char* const b) {
    size_t len_a = strlen(a), len_b = strlen(b);
    align(a, b);
    return (double)(len_a > len_b ? len_a : len_b);
}

int main(int argc, char *argv[]) {
    const char * a = "aabbccddZZ";
    const char * b = "bbFF";
    if(argc > 2) {
        a = argv[1];
        b = argv[2];
    }
    align(a, b);
    return 0;
}

void swap(int * a, int * b) {
    int tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}
