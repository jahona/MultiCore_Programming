#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.h"
#include <math.h>

void BitonicSort(int, int, bool);
void BitonicMerge(int, int, bool);
void print();
bool CheckPowOfTwo(int);

int * arr;
int * origin_arr;
int arr_length;

int main(int argc, char * argv[]) {
	arr_length = atoi(argv[1]);
	int padding = 1;
	int temp;

	if(!CheckPowOfTwo(arr_length)) {
		temp = arr_length;

		while(true) {
			temp = temp >> 1;
			padding = padding << 1;
			if(temp == 0) break;
		}

		padding -= arr_length;					
	}
	
	arr = new int[arr_length + padding];
	origin_arr = new int [arr_length + padding];

	for(int i=0 ; i<arr_length ; i++) {
		arr[i] = rand() % 200;
		origin_arr[i] = arr[i];		
	}

	for(int i=0 ; i<padding ; i++) {
		arr[arr_length + i] = INT_MAX;
		origin_arr[arr_length + i] = INT_MAX;
	}
	
	print();
		
	BitonicSort(0, arr_length+padding, true);
	
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
	for(int i=0 ; i<arr_length ; i++) {
		if(origin_arr[i] != arr[i]) {
			printf("%d*\t", arr[i]);
			origin_arr[i] = arr[i];
		} else
			printf("%d\t", arr[i]);
	}

	printf("\n");
}
	
bool CheckPowOfTwo(int num) {
	int result = num & (num - 1);

	if(result == 0)
		return true;
	else
		return false;
}
