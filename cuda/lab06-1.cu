#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.h"

#define NUM_DATA (1024 * 1024)
#define NUM_BIN (256)

#define NUM_THREADS_IN_BLOCK 1024

__global__ void globalSync(float * d_a, int * d_b, int A_SIZE)
{
	int TID = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(TID >= A_SIZE) return;

	atomicAdd(d_b + int(d_a[TID]), 1);
}

__global__ void optimizedSharedMemory(float * d_a, int * d_b, int A_SIZE)
{
	__shared__ int sh[NUM_BIN];
	int TID = blockIdx.x * blockDim.x + threadIdx.x;	

	if(threadIdx.x < NUM_BIN) sh[threadIdx.x] = 0;
	__syncthreads();

	if(TID < A_SIZE) atomicAdd(&sh[int(d_a[TID])], 1);
	__syncthreads();

	if(threadIdx.x < NUM_BIN) atomicAdd(&d_b[threadIdx.x], sh[threadIdx.x]);
}
 
void serial(float * a, int * b, int A_SIZE)
{
	for(int i=0 ; i<A_SIZE ; i++) {
		b[int(a[i])]++;
	}	
}

void check(float * a, int * b, int A_SIZE, int B_SIZE)
{
	int * temp = new int[B_SIZE]; memset(temp, 0, sizeof(int) * B_SIZE);

	for(int i=0 ; i<A_SIZE ; i++) {
		temp[int(a[i])]++;
	}
	
	bool success = true;

	for(int i=0 ; i<B_SIZE ; i++) {
		if(temp[i] != b[i]) {
			printf("index %d : result not match your value : %d, but original value : %d\n", i, b[i], temp[i]);
			success = false;
		}
	}

	if(success) {
		printf("match\n");
	}
	else {
		printf("not match\n");
	}		
}

int main() {
	DS_timer timer(8);
	timer.initTimers();

	float *a, *d_a;
	int *b, *d_b;

	int A_SIZE = NUM_DATA;
	int B_SIZE = NUM_BIN;

	int A_MEM_SIZE = A_SIZE * sizeof(float);
	int B_MEM_SIZE = B_SIZE * sizeof(int);

	a = new float[A_SIZE]; memset(a, 0, A_MEM_SIZE);
	b = new int[B_SIZE]; memset(b, 0, B_MEM_SIZE);

	for (int i = 0; i < A_SIZE; i++) {
		a[i] = rand() / (float)RAND_MAX * 256.0f; 
	}

	for (int i = 0; i < B_SIZE; i++) {
		b[i] = 0;
	}
		
	// serial
	timer.onTimer(0);
	serial(a, b, A_SIZE);
	timer.offTimer(0);

	// serial result validation check
	check(a, b, A_SIZE, B_SIZE);
	
	// initial b array	
	memset(b, 0, B_MEM_SIZE);

	// device global memory allocation
	cudaMalloc(&d_a, A_MEM_SIZE);
	cudaMalloc(&d_b, B_MEM_SIZE);

	// memory cpy Host to Device
	cudaMemcpy(d_a, a, A_MEM_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, B_MEM_SIZE, cudaMemcpyHostToDevice);

	timer.onTimer(1);
	globalSync <<< NUM_DATA / NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK >>> (d_a, d_b, A_SIZE);
	cudaThreadSynchronize();
	timer.offTimer(1);

	cudaMemcpy(b, d_b, B_MEM_SIZE, cudaMemcpyDeviceToHost);

	check(a, b, A_SIZE, B_SIZE);
	
	// optimized shared memory version start
	memset(b, 0, B_MEM_SIZE);
	
	cudaMemcpy(d_b, b, B_MEM_SIZE, cudaMemcpyHostToDevice);
	
	timer.onTimer(2);
	optimizedSharedMemory <<< NUM_DATA / NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK >>> (d_a, d_b, A_SIZE);
	timer.offTimer(2);

	cudaMemcpy(b, d_b, B_MEM_SIZE, cudaMemcpyDeviceToHost);
	
	check(a, b, A_SIZE, B_SIZE);

	// timer display
	timer.setTimerName(0, "serial");
	timer.setTimerName(1, "atomic");
	timer.setTimerName(2, "shared");

	timer.printTimer();
	
	cudaFree(d_a); cudaFree(d_b);
	
	delete[] a, delete[] b;	

	return 0;
}
