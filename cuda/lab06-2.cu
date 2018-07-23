#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.h"
#include <float.h>

#define NUM_THREADS_IN_BLOCK (1024)

#define NUM_DATA (1024 * 1024)

void serial(float start, float end, float h, int n, float * sum)
{
	float x_start, x_end, d;

	for(int i=0 ; i<n-1 ; i++) {
		x_start = start + h * i;
		x_end = start + h * (i+1);
		d = ((x_start * x_start) + (x_end * x_end)) / 2.0;
		
		*sum += d*h;
	}
}

__global__ void atomicSync(float start, float end, float h, int n, float * sum)
{
	int TID = blockIdx.x * blockDim.x + threadIdx.x; 
	if(TID == 0) {
		*sum = 0;	
	}
	__syncthreads();
	
	if(TID>=n-1) return;
	
	float x_start = start + h*TID;
	float x_end = start + h*(TID+1);
	float d = ((x_start * x_start) + (x_end * x_end)) / 2.0;
	
	atomicAdd(sum, d*h);
}

__global__ void sharedMemorySync(float start, float end, float h, int n, float * sum)
{
	int TID = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(TID >= n-1) return;
	
	__shared__ float localVal[NUM_THREADS_IN_BLOCK];
	float x_start = start + h*TID;
	float x_end = start + h*(TID+1);
	float d = ((x_start * x_start) + (x_end * x_end)) / 2.0;
	
	localVal[threadIdx.x] = d*h;
	__syncthreads();

	if(threadIdx.x == 0) {
		for(int i=1 ; i<NUM_THREADS_IN_BLOCK ; i++) {
			localVal[0] += localVal[i];
		}
	
		atomicAdd(sum, localVal[0]);
	}
}

__global__ void reduction(float start, float end, float h, int n, float * sum)
{
	int TID = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float localVal[NUM_THREADS_IN_BLOCK];

	float x_start = start + h*TID;
	float x_end = start + h*(TID+1);
	float d = ((x_start * x_start) + (x_end * x_end)) / 2.0;

	localVal[threadIdx.x] = d*h;
	__syncthreads();

	// reduction
	int offset = 1;

	while(offset < NUM_THREADS_IN_BLOCK) {
		if(threadIdx.x % (2*offset) == 0) {
			localVal[threadIdx.x] += localVal[threadIdx.x + offset];
		}

		__syncthreads();
		offset *= 2;
	}
	
	if(threadIdx.x == 0) {
		atomicAdd(sum, localVal[0]);
	}
}

__global__ void reduction2(float start, float end, float h, int n, float * sum)
{
	int TID = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float localVal[NUM_THREADS_IN_BLOCK];

	float x_start = start + h*TID;
	float x_end = start + h*(TID+1);
	float d = ((x_start * x_start) + (x_end * x_end)) / 2.0;

	localVal[threadIdx.x] = d*h;
	__syncthreads();

	// reduction
	int offset = NUM_THREADS_IN_BLOCK / 2;

	while(offset > 0) {
		if(threadIdx.x < offset) {
			localVal[threadIdx.x] += localVal[threadIdx.x + offset];
		}
		
		offset /= 2;
	
		__syncthreads();
	}
	
	if(threadIdx.x == 0) {
		atomicAdd(sum, localVal[0]);
	}
}

__device__ void warpReduce(volatile float * _localVal, int _tid)
{
	_localVal[_tid] += _localVal[_tid + 32];
	_localVal[_tid] += _localVal[_tid + 16];
	_localVal[_tid] += _localVal[_tid + 8];
	_localVal[_tid] += _localVal[_tid + 4];
	_localVal[_tid] += _localVal[_tid + 2];
	_localVal[_tid] += _localVal[_tid + 1];	
}

__global__ void reduction3(float start, float end, float h, int n, float * sum)
{
	int TID = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float localVal[NUM_THREADS_IN_BLOCK];

	float x_start = start + h*TID;
	float x_end = start + h*(TID+1);
	float d = ((x_start * x_start) + (x_end * x_end)) / 2.0;

	localVal[threadIdx.x] = d*h;
	__syncthreads();

	// reduction
	int offset = NUM_THREADS_IN_BLOCK / 2;

	while(offset > 32) {
		if(threadIdx.x < offset) {
			localVal[threadIdx.x] += localVal[threadIdx.x + offset];
		}
		
		offset /= 2;
	
		__syncthreads();
	}
	
	if(threadIdx.x < 32) warpReduce(localVal, threadIdx.x);
	__syncthreads();

	if(threadIdx.x == 0) {
		atomicAdd(sum, localVal[0]);
	}
}

void check(float start, float end, float h, int n, float * result)
{
	float x_start, x_end, d;
	float sum = 0.0;

	for(int i=0 ; i<n-1 ; i++) {
		x_start = start + h * i;
		x_end = start + h * (i+1);
		d = ((x_start * x_start) + (x_end * x_end)) / 2.0;
		
		sum += d*h;	
	}
	
	bool success = true;

	if(*result - sum >= FLT_EPSILON) {
		printf("your value is %f, but original value is %f\n", *result, sum);
		success = false;		
	}
	else if(sum - *result >= FLT_EPSILON) {
		printf("your value is %f, but original value is %f\n", *result, sum);
		success = false;
	}

	if(success) {
		printf("cpu : %f, gpu : %f match\n", sum, *result);
	}
	else {
		printf("not match\n");
	}		
}

int main() {
	DS_timer timer(8);
	timer.initTimers();
	
	float start = -10;
	float end = 10;
	
	float h = (end-start) / NUM_DATA;
	float * sum;
	float * d_sum;

	sum = new float;
	*sum = 0;
	
	// serial
	timer.onTimer(0);
	serial(start, end, h, NUM_DATA, sum);
	timer.offTimer(0);

	// serial result validation check
	check(start, end, h, NUM_DATA, sum);
	
	*sum = 0.0;

	// device global memory allocation
	cudaMalloc(&d_sum, sizeof(float));
	
	// memory cpy Host to Device
	cudaMemcpy(d_sum, sum, sizeof(float), cudaMemcpyHostToDevice);

	timer.onTimer(1);
	atomicSync <<< NUM_DATA / NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK >>> (start, end, h, NUM_DATA, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(1);

	cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

	check(start, end, h, NUM_DATA, sum);
	
	*sum = 0.0;

	cudaMemcpy(d_sum, sum, sizeof(float), cudaMemcpyHostToDevice);

	timer.onTimer(2);
	sharedMemorySync <<< NUM_DATA / NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK >>> (start, end, h, NUM_DATA, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(2);
	
	cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

	check(start, end, h, NUM_DATA, sum);
	
	*sum = 0.0;

	cudaMemcpy(d_sum, sum, sizeof(float), cudaMemcpyHostToDevice);

	timer.onTimer(3);
	reduction <<< NUM_DATA / NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK >>> (start, end, h, NUM_DATA, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(3);

	cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

	check(start, end, h, NUM_DATA, sum);

	*sum = 0.0;

	cudaMemcpy(d_sum, sum, sizeof(float), cudaMemcpyHostToDevice);

	timer.onTimer(4);
	reduction2 <<< NUM_DATA / NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK >>> (start, end, h, NUM_DATA, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(4);

	cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

	check(start, end, h, NUM_DATA, sum);

	*sum = 0.0;

	cudaMemcpy(d_sum, sum, sizeof(float), cudaMemcpyHostToDevice);

	timer.onTimer(5);
	reduction3 <<< NUM_DATA / NUM_THREADS_IN_BLOCK, NUM_THREADS_IN_BLOCK >>> (start, end, h, NUM_DATA, d_sum);
	cudaThreadSynchronize();
	timer.offTimer(5);

	cudaMemcpy(sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

	check(start, end, h, NUM_DATA, sum);

	// timer display
	timer.setTimerName(0, "serial");
	timer.setTimerName(1, "atomic");
	timer.setTimerName(2, "sharedMemory");
	timer.setTimerName(3, "reduction 1");
	timer.setTimerName(4, "reduction 2");
	timer.setTimerName(5, "reduction 3");

	timer.printTimer();
	
	cudaFree(d_sum);
	
	delete sum;

	return 0;
}
