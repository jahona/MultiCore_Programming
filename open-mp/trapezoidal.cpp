#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "DS_timer.h"
#include <iostream>
using namespace std;

#define MAX 1073741824

double f(double x) {
	return x*x;
}

void serial(double a, double b) {
	double sum = 0.0;
	double h = (b-a)/MAX;
	double x_i, x_j, d;

	for (int i=0 ; i<MAX-1 ; i++) {		
		x_i = a + h * i;
		x_j = a + h * (i+1);
		d = (f(x_i) + f(x_j)) / 2.0;

		sum += d*h;
	}

	printf("serial : %lf\n", sum);
}

void parral(double a, double b) {
	double result = 0.0;
	int count;
	double * sum;
	double h = (b-a)/MAX;

	#pragma omp parallel
	{
		#pragma omp single
		{
			count = omp_get_num_threads();
			sum = new double[omp_get_num_threads()];
		}

		int tid = omp_get_thread_num();

		#pragma omp for
		for (int i=0 ; i<MAX-1 ; i++) {		
			double x_i = a + h * i;
			double x_j = a + h * (i+1);
			double d = (f(x_i) + f(x_j)) / 2.0;

			sum[tid] += d*h;
		}
	}

	for(int i=0 ; i<count ; i++) {
		result += sum[i];
	}

	printf("parral : %lf\n", result);
}

int main() {
	int a = 0, b = 1024;

	DS_timer timer(2);

	timer.initTimers();

	timer.onTimer(0);
	serial(a, b);
	timer.offTimer(0);
	
	timer.onTimer(1);
	parral(a, b);
	timer.offTimer(1);

	timer.printTimer();

	return 0;
}