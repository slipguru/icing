#include "alignment_cpp.h"
using namespace std;

int i, j, k;

void print_alignment_matrix(int * M, int n, int m, string a, string b) {
    printf("     0");
    for(i = 0; i < m; ++i) {
    	printf("   %c", b[i]);
    }
    printf("\n");

    for(i = 0; i < n; ++i) {
    	if(i == 0) {
    		printf(" 0");
    	} else {
    		printf(" %c",a[i-1]);
        }
    	for(j = 0; j < m; ++j) {
    		printf(" %3i", M[i*m + j]);
    	}
    	printf("\n");
    }
}


int local_alignment(string a, string b, string &a_n, string &b_n) {
    /*Alignment using Smith-Waterman algorithm*/
    int len_a = a.size();
    int len_b = b.size();
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

	//print_alignment_matrix(matrix, n, m, a, b);
	/*MAKE ALIGNMENT*/
    int max_score = INT_MIN;
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

    a_n = b_n = "";
	a_n = a[imax-1];
	b_n = b[jmax-1];
    int relmax = -1;
    int relmaxpos[2];

	while(imax > -1 && jmax > -1) {
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
    		a_n = a[relmaxpos[0]-1] + a_n;
    		b_n = b[relmaxpos[1]-1] + b_n;
    	} else {
            if((relmaxpos[1] == jmax-1) && (relmaxpos[0] != imax-1)) {
                //maxB needs at least one '-'
                // value on the left
                a_n = a[imax-1] + a_n;
                b_n = GAP + b_n;
            } else if((relmaxpos[0] == imax-1) && (relmaxpos[1] != jmax-1)) {
                //MaxA needs at least one '-'
                a_n = GAP + a_n;
                b_n = b[jmax-1] + b_n;
    		}
    	}
        imax = relmaxpos[0];
        jmax = relmaxpos[1];
    }
    free(matrix);
	return 0;
}

int global_alignment(string a, string b, string &a_n, string &b_n) {
    /*Alignment using Needleman-Wunsch algorithm. */
    int len_a = a.size();
    int len_b = b.size();
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
	//print_alignment_matrix(matrix, n, m, a, b);
    int imax = len_a, jmax = len_b;

    a_n = b_n = "";
	a_n = a[imax-1];
	b_n = b[jmax-1];
	// _glob_align(matrix, max_score_i, max_score_j, n, m, a, b, a_n, b_n);

    int relmax = -1;		//hold highest value in sub columns and rows
    int relmaxpos[2];
	while(imax > -1 && jmax > -1) {
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
    		a_n = a[relmaxpos[0]-1] + a_n;
    		b_n = b[relmaxpos[1]-1] + b_n;
    	} else {
            if((relmaxpos[1] == jmax-1) && (relmaxpos[0] != imax-1)) {
                //maxB needs at least one '-'
                // value on the left
                a_n = a[imax-1] + a_n;
                b_n = GAP + b_n;
            } else if((relmaxpos[0] == imax-1) && (relmaxpos[1] != jmax-1)) {
                //MaxA needs at least one '-'
                a_n = GAP + a_n;
                b_n = b[jmax-1] + b_n;
    		}
    	}
        imax = relmaxpos[0];
        jmax = relmaxpos[1];
    }
    free(matrix);
	return 0;
}

int main() {
    string a = "AAATGCGGAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    string b = "ATGCAAA";
    string c, d;
    // local_alignment(a, b, c, d);
    global_alignment(a, b, c, d);
	cout << c << '\n' << d << endl;
}
