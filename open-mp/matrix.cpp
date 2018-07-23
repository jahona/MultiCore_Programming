#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"
#include <iostream>
using namespace std;

#define MAX 10000

float A[MAX][MAX];
float X[MAX];
float Y[MAX];

void inputRand() {
	for(int i=0 ; i<MAX ; i++) {
		X[i] = rand() % 100;
		for(int j=0 ; j<MAX ; j++) {
			A[i][j] = rand() % 100;
		}
	}
}

void serial() {
	for(int i=0 ; i<MAX ; i++) {
		Y[i] = 0.0;
		for(int j=0 ; j<MAX ; j++) {
			Y[i] += A[i][j] * X[j];
		}
	}
}

void parral() {
	#pragma omp parallel
	#pragma omp for
	for(int i=0 ; i<MAX ; i++) {
		Y[i] = 0.0;
		for(int j=0 ; j<MAX ; j++) {
			Y[i] += A[i][j] * X[j];
		}
	}

}


int main() {
	DS_timer timer(2); // 4개의 타이머 생성

	inputRand();
	
	timer.initTimers();

	timer.onTimer(0);
	serial();
	timer.offTimer(0);

	timer.onTimer(1);
	parral();
	timer.offTimer(1);
	timer.printTimer();

	return 0;
}

