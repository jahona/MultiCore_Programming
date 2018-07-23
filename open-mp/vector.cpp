#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"

#define MAX 1024 * 1024 * 128

int A[MAX], B[MAX], C[MAX];

void inputRandNumber(int * arr) {
	for(int i=0 ; i<MAX ; i++) {
		arr[i] = rand() % 10;
	}
}

void getSum() {
	for(int i=0; i<MAX ; i++) {
		C[i] = A[i] + B[i];
	}
}

void getSumForParral(int st, int end) {
	printf("%d %d %d\n", omp_get_thread_num(), st, end);
	for(int i=st; i<end ; i++) {
		C[i] = A[i] + B[i];
	}
}


int main(int argc, char * argv[]) {
	inputRandNumber(A);
	inputRandNumber(B);

	int p = atoi(argv[1]);
	DS_timer timer(2);
	timer.initTimers();
	
	timer.onTimer(0);
	getSum();
	timer.offTimer(0);
	
	timer.onTimer(1);

#pragma omp parallel num_threads(p)
	getSumForParral(MAX/p * omp_get_thread_num(), (MAX/p ) * (omp_get_thread_num() + 1));

	timer.offTimer(1);

	timer.printTimer();
	return 0;
}