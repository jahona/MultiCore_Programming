#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.h"

#pragma warning(disable : 4996)

#define BLOCK_SIZE (8)

__global__ void shared_matrix_multi(float *_a, float *_b, float *_c, int m, int n, int k)
{
	__shared__ float sub_a[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sub_b[BLOCK_SIZE * BLOCK_SIZE];

	int dx = blockIdx.x * blockDim.x + threadIdx.x;
	int dy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int index;
	int A_SIZE = m * k;
	int B_SIZE = k * n;

	float temp = 0.;

	// load sub matrix from global memory to shared memory
	for(int sub = 0 ; sub < gridDim.x ; sub++) {
		index = dy * k + sub * BLOCK_SIZE + threadIdx.x;

		if(index >= A_SIZE) {
			sub_a[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
		}
		else {
			sub_a[threadIdx.y * BLOCK_SIZE + threadIdx.x] = _a[index];
		}

		index = (sub * BLOCK_SIZE + threadIdx.y) * n + dx;
		
		/*
		공유 데이터 배열에 넣기 전에 전치를 시킨다.
		*/
		if(index >= B_SIZE) {
			sub_b[threadIdx.x * BLOCK_SIZE + threadIdx.y] = 0;
		}
		else {
			sub_b[threadIdx.x * BLOCK_SIZE + threadIdx.y] = _b[index];
		}
					
		__syncthreads();

		// compute value
		for(int k=0 ; k<BLOCK_SIZE ; k++) {
			// temp += sub_a[threadIdx.y][k] + sub_b[k][threadIdx.x]
			/*
			B배열에 해당하는 공유 메모리를 전치시켰기 때문에 인덱스 부분도 고쳐준다.
			*/
			temp += sub_a[threadIdx.y * BLOCK_SIZE + k] * sub_b[threadIdx.x * BLOCK_SIZE + k];
		}

		__syncthreads();
	}

	if(dy < m && dx < n) {
		_c[dy * n + dx] = temp;
	}		
}

void serial(float *a, float *b, float *c, int m, int n, int k)
{
	for (int index_m = 0; index_m < m; index_m++) {
		for (int index_n = 0; index_n < n; index_n++) {
			c[index_m * n + index_n] = 0.0;
			for (int index_k = 0; index_k < k; index_k++) {
				c[index_m * n + index_n] += a[index_m * k + index_k] * b[index_k * n + index_n];
			}
		}
	}
}

void check(float *a, float *b, float *c, int m, int n, int k, int C_SIZE)
{
	bool result = true;

	float * temp = new float[C_SIZE];

	for (int index_m = 0; index_m < m; index_m++) {
		for (int index_n = 0; index_n < n; index_n++) {
			temp[index_m * n + index_n] = 0.0;
			for (int index_k = 0; index_k < k; index_k++) {
				temp[index_m * n + index_n] += a[index_m * k + index_k] * b[index_k * n + index_n];
			}
		}
	}

	for (int i = 0; i < C_SIZE; i++) {
		if (temp[i] - c[i] > 1.0) {
			printf("[%d] The resutls is not matched! (c: %.2f, temp: %.2f)\n", i, c[i], temp[i]);
			result = false;
		}

		if (c[i] - temp[i] > 1.0) {
			printf("[%d] The resutls is not matched! (c: %.2f, temp: %.2f)\n", i, c[i], temp[i]);
			result = false;
		}
	}

	if (result)
		printf("works well!\n");
	else
		printf("not work\n");
}

int main() {
	DS_timer timer(4);
	timer.initTimers();

	int m, n, k;

	printf("m, n, k=");
	scanf("%d %d %d", &m, &n, &k);

	float *a, *b, *c;
	float *d_a, *d_b, *d_c;

	int A_SIZE = m * k;
	int B_SIZE = k * n;
	int C_SIZE = m * n;

	int A_MEM_SIZE = A_SIZE * sizeof(float);
	int B_MEM_SIZE = B_SIZE * sizeof(float);
	int C_MEM_SIZE = C_SIZE * sizeof(float);

	a = new float[A_SIZE]; memset(a, 0, A_MEM_SIZE);
	b = new float[B_SIZE]; memset(b, 0, B_MEM_SIZE);
	c = new float[C_SIZE]; memset(c, 0, C_MEM_SIZE);

	for (int i = 0; i < A_SIZE; i++) {
		a[i] = (rand() % 100) / 10.0 + 1; 
	}

	for (int i = 0; i < B_SIZE; i++) {
		b[i] = (rand() % 100) / 10.0 + 1;
	}

	// serial
	timer.onTimer(0);
	serial(a, b, c, m, n, k);
	timer.offTimer(0);

	// serial result validation check
	check(a, b, c, m, n, k, C_SIZE);
	
	// device global memory allocation
	timer.onTimer(1);
	cudaMalloc(&d_a, A_MEM_SIZE);
	cudaMalloc(&d_b, B_MEM_SIZE);
	cudaMalloc(&d_c, C_MEM_SIZE);
	
	// memory cpy Host to Device
	cudaMemcpy(d_a, a, A_MEM_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, B_MEM_SIZE, cudaMemcpyHostToDevice);
	timer.offTimer(1);
	
	int grid_N = n;
	int grid_M = m;

	if(k*k > n*m) {
		grid_N = k;
	}
		
	dim3 dimGrid((grid_N-1) / BLOCK_SIZE + 1, ((grid_M-1)/BLOCK_SIZE + 1));
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	
	timer.onTimer(2);
	shared_matrix_multi <<< dimGrid, dimBlock >>> (d_a, d_b, d_c, m, n, k);
	cudaThreadSynchronize();
	timer.offTimer(2);

	timer.onTimer(3);
	cudaMemcpy(c, d_c, C_MEM_SIZE, cudaMemcpyDeviceToHost);
	timer.offTimer(3);

	check(a, b, c, m, n, k, C_SIZE);
	
	// timer display
	timer.printTimer();
	
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	delete[] a, delete[] b, delete[] c;	

	return 0;
}
