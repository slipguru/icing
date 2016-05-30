#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <climits>

#define TRUE 1
#define FALSE 0
#define Match 5
#define MissMatch -4
#define GapPenalty 10
#define GapExt 8

void print_alignment_matrix(int * M, int n, int m, string a, string b);
int local_alignment(string a, string b, string &a_n, string &b_n);
int global_alignment(string a, string b, string &a_n, string &b_n);
