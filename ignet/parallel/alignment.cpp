

//This program creates an optimal local alignment of two sequences of symbols
//(<20 characters in length).Using the Smith-Waterman Algorithm, all possible
//alignments are scored and an optimal sequence is returned to the user.

#include <stdio.h>
#include <stdlib.h>
// #include <string.h>
#include <iostream>
#include <string>
using namespace std;
/*File pointers*/
//FILE *ptr_file_1, *ptr_file_2;	//test file pointers

/*Definitions*/
#define TRUE 1
#define FALSE 0
#define Match 5
#define MissMatch -4
#define GapPenalty 10
#define GapExt 8

/*Global Variables*/
//char inputC[5];		//user input character
//int inputI;
//int StrLen1,StrLen2;
//int intcheck = TRUE;

char holder, ch;
int filelen1 = 0;
int filelen2 = 0;
int i,j,k,l,m,n,lenA,lenB,compval;
char dash = '-';

// char strA[20];			//holds 1st string to be aligned in character array
// char strB[20];			//holds 2nd string to be aligned in character array
int HiScore;			//holds value of highest scoring alignment(s).
int HiScorePos[2];		//holds the position of the HiScore
//int SWArray[21][21];	//S-W Matrix
int ** SWArray;
// char MaxA[20];
// char MaxB[20];
// char OptA[20];
// char OptB[20];
string MaxA, MaxB, OptA, OptB, strA, strB;

int MaxAcounter = 1;	//MaxA counter
int MaxBcounter = 1;	//MaxB counter
int cont = TRUE;
int check;


/*ALIGNMENT FUNCTION*/
int Align(int PosA, int PosB) {
	/*Function Variables*/
	int relmax = -1;		//hold highest value in sub columns and rows
	int relmaxpos[2];		//holds position of relmax

	if(SWArray[PosA-1][PosB-1] == 0 && SWArray[PosA][PosB-1] == 0 && SWArray[PosA-1][PosB] == 0) {
		cont = FALSE;
	}

	while(cont == TRUE) {	//until the diagonal of the current cell has a value of zero
		/*Find relmax in sub columns and rows*/
        relmax = SWArray[PosA-1][PosB-1];
        relmaxpos[0] = PosA-1;
        relmaxpos[1] = PosB-1;
        if(relmax < SWArray[PosA-1][PosB]) {
            relmax = SWArray[PosA-1][PosB];
            relmaxpos[0] = PosA-1;
            relmaxpos[1] = PosB;
        }
        if(relmax < SWArray[PosA][PosB-1]) {
            relmax = SWArray[PosA][PosB-1];
            relmaxpos[0] = PosA;
            relmaxpos[1] = PosB-1;
        }

		// for(i=PosA; i>0; --i) {
		// 	if(relmax < SWArray[i-1][PosB-1]) {
		// 		relmax = SWArray[i-1][PosB-1];
		// 		relmaxpos[0]=i-1;
		// 		relmaxpos[1]=PosB-1;
		// 	}
		// }
        //
		// for(j=PosB; j>0; --j) {
		// 	if(relmax < SWArray[PosA-1][j-1]) {
		// 		relmax = SWArray[PosA-1][j-1];
		// 		relmaxpos[0]=PosA-1;
		// 		relmaxpos[1]=j-1;
		// 	}
		// }

		/*Align strings to relmax*/
		if((relmaxpos[0] == PosA-1) && (relmaxpos[1] == PosB-1)) {	//if relmax position is diagonal from current position simply align and increment counters
			MaxA = MaxA + strA[relmaxpos[0]-1];
			MaxB = MaxB + strB[relmaxpos[1]-1];
		} else {
			if((relmaxpos[1] == PosB-1) && (relmaxpos[0] != PosA-1)) {	//maxB needs at least one '-'
				for(i=PosA-1; i>relmaxpos[0]-1; --i) {	//for all elements of strA between PosA and relmaxpos[0]
						MaxA = MaxA + strA[i-1];
                        MaxB = MaxB + dash;
				}
				// for(j=PosA-1; j>relmaxpos[0]; --j) {	//set dashes to MaxB up to relmax
				// }
				MaxB = MaxB + strB[relmaxpos[1]-1];	//at relmax set pertinent strB value to MaxB
			} else if((relmaxpos[0] == PosA-1) && (relmaxpos[1] != PosB-1)) {	//MaxA needs at least one '-'
				for(j=PosB-1; j>relmaxpos[1]-1; --j) {	//for all elements of strB between PosB and relmaxpos[1]
					MaxB = MaxB + strB[j-1];
                    MaxA = MaxA + dash;
				}
				// for(i=PosB-1; i>relmaxpos[1]; --i) {		//set dashes to strA
				// }
				MaxA = MaxA + strA[relmaxpos[0]-1];
			}
		}
		Align(relmaxpos[0], relmaxpos[1]);
	}

	return(cont);
}

int alignment(string a, string b) {
    strA = a;
    strB = b;
    lenA = strA.size();
    lenB = strB.size();

	//Create empty table
    SWArray = new int*[lenA+1];
    SWArray[0] = new int[(lenA+1) * (lenB+1)];
    for (i = 1; i < lenA+1; ++i)
        SWArray[i] = SWArray[0] + i * (lenB+1);

	for(i=0; i<=lenA; ++i){
		SWArray[0][i] = 0;
	}
	for(i=0; i<=lenB; ++i){
		SWArray[i][0] = 0;
	}

	//score table with S-W
	for(i = 1; i <= lenA; ++i) {	//for all values of strA
		for(j = 1; j <= lenB; ++j) {	//for all values of strB
            compval = 0;
			if(strA[i-1] == strB[j-1]) { //match
				compval = (SWArray[i-1][j-1] + Match);	//compval = diagonal + match score
			} else { //missmatch
                compval = (SWArray[i-1][j-1] + MissMatch);
            }

			for(k = i-1; k > 0; --k) {		//check all sub rows
				if(compval < ((SWArray[i-k][j]) - (GapPenalty + (GapExt * k)))) {	    //if cell above has a greater value
					compval = ((SWArray[i-k][j]) - (GapPenalty + (GapExt * k)));		//set compval to that square
				}
			}

			for(k = j-1; k > 0; --k) {		//check all sub columns
				if(compval < ((SWArray[i][j-k]) - (GapPenalty + (GapExt * k)))) {	//if square to the left has the highest value
					compval = ((SWArray[i][j-k]) - (GapPenalty + (GapExt * k)));    //set compval to that square
				}
			}

			if(compval < 0) {
				compval = 0;
			}

			SWArray[i][j] = compval;	//set current cell to highest possible score and move on
		}
	}

	/*PRINT S-W Table*/
	printf("   0");
	for(i = 0; i <= lenB; ++i) {
		printf("  %c",strB[i]);
	}
	printf("\n");

	for(i = 0; i <= lenA; ++i) {
		if(i < 1) {
			printf("0");
		}

		if(i > 0) {
			printf("%c",strA[i-1]);
		}

		for(j = 0; j <= lenB; ++j) {
			printf("%3i",SWArray[i][j]);
		}
		printf("\n");
	}

	/*MAKE ALIGNMENTT*/
	for(i=0; i<=lenA; ++i) {	//find highest score in matrix: this is the starting point of an optimal local alignment
		for(j=0; j<=lenB; ++j) {
			if(SWArray[i][j] > HiScore) {
				HiScore = SWArray[i][j];
				HiScorePos[0]=i;
				HiScorePos[1]=j;
			}
		}
	}

	/*send Position to alignment function*/
    MaxA = MaxB = "";
	MaxA = strA[HiScorePos[0]-1];
	MaxB = strB[HiScorePos[1]-1];

	check = Align(HiScorePos[0], HiScorePos[1]);

	/*in the end reverse Max A and B*/
	k=0;
    OptA = MaxA;
	for(i=(MaxA.size())-1; i > -1; --i) {
		OptA[k] = MaxA[i];
		++k;
	}

	k=0;
    OptB = MaxB;
	for(j=(MaxB.size())-1; j > -1; --j) {
		OptB[k] = MaxB[j];
		++k;
	}
//	printf("\n%s\n%s\n",MaxA,MaxB);
	cout << OptA << '\n' << OptB << endl;
	return(0);

}

int main() {
    return alignment("AAATGC", "ATGC");
}
