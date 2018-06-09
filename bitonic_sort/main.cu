#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.h"

void BitonicSort(int, int, bool);
void BitonicMerge(int, int, bool);
void print();

int * arr;
int * origin_arr;
int MAX;

int main(int argc, char * argv[]) {
	MAX = atoi(argv[1]);

	arr = new int[MAX];
	origin_arr = new int [MAX];

	for(int i=0 ; i<MAX ; i++) {
		arr[i] = rand() % 200;
		origin_arr[i] = arr[i];		
	}

	print();

	BitonicSort(0, MAX, true);

	print();
}

void BitonicSort(int start, int n, bool isAscending) {
	if(n==0) return;

	int m = n/2;

	BitonicSort(start, m, true);
	BitonicSort(start+m, m, false);

	BitonicMerge(start, n, isAscending);
}

void BitonicMerge(int start, int n, bool isAscending) {
	if(n==1) return;

	int m = n/2;
	int temp;

	for(int i=start ; i<start+m ; i++) {
		if(isAscending == (arr[i] > arr[i+m])) {
			temp = arr[i];
			arr[i] = arr[i+m];
			arr[i+m] = temp;
			print();
		}
	}

	BitonicMerge(start, m, isAscending);
	BitonicMerge(start+m, m, isAscending);
}

void print() {
	for(int i=0 ; i<MAX ; i++) {
		if(origin_arr[i] != arr[i]) {
			printf("%d*\t", arr[i]);
			origin_arr[i] = arr[i];
		} else
			printf("%d\t", arr[i]);
	}

	printf("\n");
}
	
