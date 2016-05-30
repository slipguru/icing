int global_alignment(char * a, char * b, char ** a_n, char ** b_n) {
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

    (*a_n) = malloc(sizeof(char)*(len_a+len_b+1)); // need space for '\0'
    (*b_n) = malloc(sizeof(char)*(len_a+len_b+1)); // need space for '\0'
    if((*a_n) == NULL || (*b_n) == NULL) {
		fprintf(stderr, "Error, malloc failed");
	}
    //
    // strcpy(out_a, a[imax-1]);
    // strcpy((*a_n), a+n-2);
    // strcpy(out_b, b[imax-1]);

    (*a_n)[1] = '\0';
    (*a_n)[0] = a[imax-1];
    (*b_n)[1] = '\0';
    (*b_n)[0] = b[jmax-1];

    // fprintf(stderr, "a = %s\n", a);
    // fprintf(stderr, "b = %s\n", b);

    int relmax = -1;
    int relmaxpos[2];
    // int pos_a = 1;
    int cont = 0;
	while(imax > 0 && jmax > 0) {
        cont++;
        fprintf(stderr, "a_n = %s\n", (*a_n));
        fprintf(stderr, "b_n = %s\n", (*b_n));
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

        // fprintf(stderr, "i = %d, j = %d, reli = %d, relj = %d\n", imax,jmax,relmaxpos[0],relmaxpos[1]);
    	if((relmaxpos[0] == imax-1) && (relmaxpos[1] == jmax-1)) {
            //if relmax position is diagonal from current position simply align
            (*a_n)[cont] = a[relmaxpos[0] - 1];
            (*a_n)[cont+1] = '\0';

    		(*b_n)[cont] = b[relmaxpos[1] - 1];
            (*b_n)[cont+1] = '\0';
    	} else {
            if((relmaxpos[1] == jmax-1) && (relmaxpos[0] != imax-1)) {
                //maxB needs at least one '-'
                // value on the left
                (*a_n)[cont] = a[imax - 1];
                (*a_n)[cont+1] = '\0';

        		(*b_n)[cont] = dash;
                (*b_n)[cont+1] = '\0';
            } else if((relmaxpos[0] == imax-1) && (relmaxpos[1] != jmax-1)) {
                //MaxA needs at least one '-'
                (*a_n)[cont] = dash; //a + relmaxpos[0] - 1;
                (*a_n)[cont+1] = '\0';

        		(*b_n)[cont] = b[jmax - 1];
                (*b_n)[cont+1] = '\0';
    		}
    	}
        imax = relmaxpos[0];
        jmax = relmaxpos[1];
    }
    free(matrix);

    /*in the end reverse Max A and B*/
    char * _tmp = malloc(sizeof(char)*(len_a+len_b+1));
	for(i = strlen((*a_n))-1, k = 0, strcpy(_tmp, (*a_n)); i > -1; --i, ++k) {
		_tmp[k] = (*a_n)[i];
	}
    strcpy((*a_n), _tmp);
	for(j = strlen((*b_n))-1, k = 0, strcpy(_tmp, (*b_n)); j > -1; --j, ++k) {
		_tmp[k] = (*b_n)[j];
	}
    strcpy((*b_n), _tmp);
    free(_tmp);

    fprintf(stderr, "a_n = %s\n", (*a_n));
    fprintf(stderr, "b_n = %s\n", (*b_n));

	return 0;
}
