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

void print_alignment_matrix(int * M, int n, int m, const char * a, const char * b) {
    printf("  ");
    for(j = 0; j < m; ++j) {
    	printf("   %c", b[j]);
    }
    printf("\n");
    for(i = 0; i < n; ++i, printf("\n")) {
        printf(" %c", a[i]);
    	for(j = 0; j < m; ++j) {
    		printf(" %3i", M[i*m + j]);
    	}
    }
    printf("\n");
}

#define MATCH 1
#define MISMATCH 0
int match_fn(const char a, const char b) {
    return (a == b) ? MATCH : MISMATCH;
}

int global_alignment(const char * const a, const char * const b, char * a_n, char * b_n) {
    /*Alignment using Needleman-Wunsch algorithm. */
    int len_a = strlen(a);
    int len_b = strlen(b);
    int n = len_a;
    int m = len_b;

	//Create empty table
    int * matrix = (int *)malloc(n*m*sizeof(int));
	//matrix[0] = 0;
	for(j = 0; j < m; ++j) { // initialise first row
		matrix[j] = match_fn(a[0], b[j]);
	}
	for(i = 0; i < n; ++i) { // initialise first col
		matrix[i*m] = match_fn(a[i], b[0]);
	}
    print_alignment_matrix(matrix, n, m, a, b);

    int best_score, best_indexi, best_indexj;
    int score;
	for(int row = 1; row < n; ++row) {
		for(int col = 1; col < m; ++col) {
            best_score = matrix[(row-1)*m + col-1];
            best_indexi = row-1;
            best_indexj = col-1;

            for(i = 0; i < col-1; ++i) {
                score = matrix[(row-1)*m + i];
                if(score > best_score) {
                    best_score = score;
                    best_indexi = row-1;
                    best_indexj = i;
                }
            }
            for(i = 0; i < row-1; ++i) {
                score = matrix[i*m+col-1];
                if(score > best_score) {
                    best_score = score;
                    best_indexi = i;
                    best_indexj = col-1;
                }
            }
			matrix[row*m + col] = best_score + match_fn(a[row], b[col]);
		}
	}

    // Find global start
    // 1. Search all rows in the last column.
    best_score = matrix[(n-1)*m + m-1]; // initialise with last element
    best_indexi = n-1;
    best_indexj = m-1;
    for(int row = 0; row < n; ++row) {
        score = matrix[(row)*m + m-1];
        if(score > best_score) {
            best_score = score;
            best_indexi = row;
            best_indexj = m-1;
        }
    }
    // 2. Search all columns in the last row.
    for(int col = 0; col < m; ++col) {
        score = matrix[(n-1)*m + col];
        if(score > best_score) {
            best_score = score;
            best_indexi = n-1;
            best_indexj = col;
        }
    }

    // seqA, seqB, score, begin, end, prev_pos, next_pos
    // a, b, best_score, None, None, (n, m), (best_indexi, best_indexj)
    // prevA == best_indexi, prevB == best_indexj
    while(imax > 0 && jmax > 0) {
        // TODO completare con recover_alignments, in pairwise2
    }

    int imax = len_a, jmax = len_b;
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
    if(argc > 2) {
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
