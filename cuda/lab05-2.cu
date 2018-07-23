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

		/*
		A배열에서 dy에 해당하는 행에서 시작하는 열들을 공유메모리에 넣어준다.
		만약 A_SIZE보다 큰 경우는 A배열을 벗어나기 때문에 예외처리를 해준다.
		*/
		index = dy * k + sub * BLOCK_SIZE + threadIdx.x;

		if(index >= A_SIZE) {
			sub_a[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
		}
		else {
			sub_a[threadIdx.y * BLOCK_SIZE + threadIdx.x] = _a[index];
		}

		/*
		B배열에서 dx에 해당하는 열에서 시작하는 행들을 공유메모리에 넣어준다.
		만약 B_SIZE보다 큰 경우는 B배열을 벗어나기 때문에 예외처리를 해준다.
		*/
		index = (sub * BLOCK_SIZE + threadIdx.y) * n + dx;
		
		if(index >= B_SIZE) {
			sub_b[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0;
		}
		else {
			sub_b[threadIdx.y * BLOCK_SIZE + threadIdx.x] = _b[index];
		}
					
		__syncthreads();	

		// compute value
		for(int k=0 ; k<BLOCK_SIZE ; k++) {
			// temp += sub_a[threadIdx.y][k] + sub_b[k][threadIdx.x]
			temp += sub_a[threadIdx.y * BLOCK_SIZE + k] * sub_b[BLOCK_SIZE * k + threadIdx.x];
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

	/*
	현재 구현한 방식이 shared_matrix_multi에서 공유 메모리에 올리기 위해 데이터를 분할하는 작업이 필요한데,
	데이터 분할을 위해 A배열은 가로 데이터를, B배열은 세로 데이터를 계속 넣어줘야 한다.
	코드 단순화를 위해 목적 배열인 C 배열을 기준으로 세로는 m 만큼, 가로는 n 만큼 인덱스를 무작위 생성하여 필요한 인덱스만 뽑아쓰는 방식을 취하고 있습니다.
	
	그래서 행열들의 크기는 아래와 같은 조건을 만족해야 합니다.

	m*n > m*k
	m*n > n*k
	
	만약 위 조건을 만족하지 않으면, 인덱스를 구하기 위해 필요한 쓰레드의 수가 부족하기 때문에 쓰레드의 수를 일정 부분 늘려주는 방식을 채택하였습니다.
	(이론이 잘 맞는지는 모르겠습니다.)
	*/
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
