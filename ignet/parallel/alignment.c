/* Alignment module  */
#include "alignment.h"

/*Definitions*/
int i, j, k;

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
    int max_score = INT_MIN;
    int imax, jmax;
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
	// print_alignment_matrix(matrix, n, m, a, b);
    int imax = len_a, jmax = len_b;

    // a_n = malloc(sizeof(char)*(len_a+len_b+1)); // need space for '\0'
    // b_n = malloc(sizeof(char)*(len_a+len_b+1)); // need space for '\0'
    // if(a_n == NULL || b_n == NULL) {
	// 	fprintf(stderr, "Error, malloc failed");
	// }
    //
    // strcpy(out_a, a[imax-1]);
    // strcpy(a_n, a+n-2);
    // strcpy(out_b, b[imax-1]);
    print_alignment_matrix(matrix, n, m, a, b);

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


double cdist_function(const char * a, const char * b) {
    size_t len_a = strlen(a), len_b = strlen(b);
    char * c = (char *)malloc(sizeof(char)*(len_a+len_b+1));
    char * d = (char *)malloc(sizeof(char)*(len_a+len_b+1));
    if(!c || !d) {
        fprintf(stderr, "Error, malloc failed");
    }
    global_alignment(a, b, c, d);
    fprintf(stderr, "EHI!  %s and %s\n", c, d);
    free(c);
    free(d);
    return (double)(len_a > len_b ? len_a : len_b);
}

int main(int argc, char *argv[]) {
    const char * a = "aabbccddZZ-";
    const char * b = "bbFF-";
    if(argc > 1) {
        a = argv[1];
        b = argv[2];
    }
    size_t len_a = strlen(a), len_b = strlen(b);
    char * c = malloc(sizeof(char)*(len_a+len_b+1));
    char * d = malloc(sizeof(char)*(len_a+len_b+1));
    if(c == NULL || d == NULL) {
		fprintf(stderr, "Error, malloc failed");
	}

    // local_alignment(a, b, c, d);
    global_alignment(a, b, c, d);
	printf("%s\n%s\n", c, d);
    free(c);
    free(d);
    return 0;
}

void swap(int * a, int * b) {
    int tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}
