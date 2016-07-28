/* author: Federico Tomasi
 * license: FreeBSD License
 * copyright: Copyright (C) 2016 Federico Tomasi
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <bits/types.h> // needed to efficient reversing strings
#include <stdio.h>

#define TRUE 1
#define FALSE 0
// #define Match 5
// #define MissMatch -5
// #define GapPenalty 10
// #define GapExt 8
#define Match 1
#define MissMatch -1
#define GapPenalty 0
#define GapExt 0
#define GAP (char)'-'
#define SWP(x,y) (x^=y, y^=x, x^=y)

// macro for globalxx
#define MATCH 1
#define MISMATCH 0

static void print_alignment_matrix(int * M, int n, int m, const char * a, const char * b);
int local_alignment(const char * a, const char * b, char * a_n, char * b_n);
int global_alignment(const char * a, const char * b, char * a_n, char * b_n);
static int globalxx(const char * a, const char * b, char * a_n, char * b_n);
double cdist_function(const char * a, const char * b);
