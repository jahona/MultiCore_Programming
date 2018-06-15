#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DS_timer.h"
#include <math.h>

#define CPU_RECURSIVE_VERSION
#define CUDA_NORMAL_SEPERATE_MODE
#define CUDA_SHARED_MEMORY_MODE	

int * result_arr;
int * arr;
int * origin_arr;
int arr_length = 1024 * 1024 * 16 + 5;
void print();
bool CheckPowOfTwo(int num);
void check();

#ifdef CUDA_SHARED_MEMORY_MODE	
__global__ void bitonic_in_block_shared(int *d_arr) {
   int TID = blockIdx.y * (gridDim.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockDim.x * threadIdx.y) + threadIdx.x;
   int half_of_stride, flag, offset, temp, target;

   __shared__ int sh[1024];

   int TID_IN_BLOCK = (blockDim.x * threadIdx.y) + threadIdx.x;

   sh[TID_IN_BLOCK] = d_arr[TID];
   __syncthreads();

   for (int size = 2; size <= blockDim.x * blockDim.y; size = size << 1) {
      flag = 0;
      for (int stride = size; stride > 1; stride = stride >> 1) {
         half_of_stride = stride >> 1;
         offset = TID % stride;
         if (offset < half_of_stride) {
            if (flag == 0) {
               target = TID_IN_BLOCK + stride - 1 - 2 * offset;
            }
            else {
               target = TID_IN_BLOCK + half_of_stride;
            }

            if (sh[TID_IN_BLOCK] > sh[target]) {
               temp = sh[target];
               sh[target] = sh[TID_IN_BLOCK];
               sh[TID_IN_BLOCK] = temp;
            }
         }

         __syncthreads();
         flag = 1;
      }
   }

   d_arr[TID] = sh[TID_IN_BLOCK];
}

__global__ void bitonic_merge_between_block_shared(int *d_arr, int stride, int flag) {
   int TID = blockIdx.y * (gridDim.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockDim.x * threadIdx.y) + threadIdx.x;
   int target, temp;
   int offset = TID % stride;
   int half_of_stride = stride >> 1;

   if (offset < half_of_stride) {
      if (flag == 0) {
         target = TID + stride - 1 - 2 * offset;
      }
      else {
         target = TID + half_of_stride;
      }

      if (d_arr[TID] > d_arr[target]) {
         temp = d_arr[target];
         d_arr[target] = d_arr[TID];
         d_arr[TID] = temp;
      }
   }
}

__global__ void bitonic_merge_in_block_shared(int * d_arr, int stride) {
   int TID = blockIdx.y * (gridDim.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockDim.x * threadIdx.y) + threadIdx.x;
   int half_of_stride, target, temp;

   __shared__ int sh[1024];
  
   int TID_IN_BLOCK = blockDim.x * threadIdx.y + threadIdx.x;

   sh[TID_IN_BLOCK] = d_arr[TID];
   __syncthreads();

   while (stride > 1) {
      half_of_stride = stride >> 1;

      if (TID % stride < half_of_stride) {
         target = TID_IN_BLOCK + half_of_stride;

         if (sh[TID_IN_BLOCK] > sh[target]) {
            temp = sh[target];
            sh[target] = sh[TID_IN_BLOCK];
            sh[TID_IN_BLOCK] = temp;
         }
      }

      __syncthreads();

      stride = stride >> 1;
   }

   d_arr[TID] = sh[TID_IN_BLOCK];
}

void bitonic_sort_from_host_shared(int n, int dx, int dy, int block_size) {
    int num_threads_in_block = (block_size * block_size);

    int * d_arr;

    cudaMalloc(&d_arr, sizeof(int) * n);
    cudaMemcpy(d_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice);

    int flag;
    dim3 dimGrid(dx / block_size, dy / block_size);
    dim3 dimBlock(block_size, block_size);

    bitonic_in_block_shared << <dimGrid, dimBlock >> > (d_arr);

    for (int size = 2 * num_threads_in_block ; size <= n; size = size << 1) {
        flag = 0;

        for (int stride = size; stride > num_threads_in_block; stride >>= 1) {
            bitonic_merge_between_block_shared << <dimGrid, dimBlock >> > (d_arr, stride, flag);
            flag = 1;
        }

        bitonic_merge_in_block_shared << < dimGrid, dimBlock >> > (d_arr, num_threads_in_block);
    }

    cudaMemcpy(arr, d_arr, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

#endif


#ifdef CUDA_NORMAL_SEPERATE_MODE

__global__ void bitonic_in_block(int *d_arr) {
    int TID = blockIdx.y * (gridDim.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockDim.x * threadIdx.y) + threadIdx.x;
    int half_of_stride, flag, offset, temp, target;

    for (int size = 2; size <= blockDim.x * blockDim.y ; size = size << 1) {
        flag = 0;

        for (int stride = size; stride > 1; stride = stride >> 1) {
            half_of_stride = stride >> 1;

            offset = TID % stride;

            if (offset < half_of_stride) {
                if (flag == 0) {
                    target = TID + stride - 1 - 2 * offset;
                }

                else {
                    target = TID + half_of_stride;
                }

                if (d_arr[TID] > d_arr[target]) {
                    temp = d_arr[target];
                    d_arr[target] = d_arr[TID];
                    d_arr[TID] = temp;
                }
            }
            __syncthreads();
            flag = 1;
        }
    }
}

__global__ void bitonic_merge_between_block(int *d_arr, int stride, int flag) {
    int TID = blockIdx.y * (gridDim.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockDim.x * threadIdx.y) + threadIdx.x;
    int target, temp;
    int offset = TID % stride;
    int half_of_stride = stride >> 1;

    if (offset < half_of_stride) {
        if (flag == 0) {
            target = TID + stride - 1 - 2*offset;
        }
        else {
            target = TID + half_of_stride;
        }

        if (d_arr[TID] > d_arr[target]) {
            temp = d_arr[target];
            d_arr[target] = d_arr[TID];
            d_arr[TID] = temp;
        }
    }
}

__global__ void bitonic_merge_in_block(int * d_arr, int stride) {
    int TID = blockIdx.y * (gridDim.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)) + (blockDim.x * threadIdx.y) + threadIdx.x;

    int half_of_stride, target, temp;

    while (stride > 1) {
        half_of_stride = stride >> 1;

        if (TID % stride < half_of_stride) {
            target = TID + half_of_stride;

            if (d_arr[TID] > d_arr[target]) {
                temp = d_arr[target];
                d_arr[target] = d_arr[TID];
                d_arr[TID] = temp;
            }
        }

        __syncthreads();

        stride = stride >> 1;
    }
}

void bitonic_sort_from_host(int n, int dx, int dy, int block_size) {
	int num_threads_in_block = (block_size * block_size);

    int * d_arr;

    cudaMalloc(&d_arr, sizeof(int) * n);
    cudaMemcpy(d_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice);

    int flag;
    dim3 dimGrid(dx / block_size, dy / block_size);
    dim3 dimBlock(block_size, block_size);

    bitonic_in_block << <dimGrid, dimBlock >> > (d_arr);

    for (int size = 2 * num_threads_in_block ; size <= n; size = size << 1) {
        flag = 0;

        for (int stride = size; stride > num_threads_in_block; stride >>= 1) {
            bitonic_merge_between_block << <dimGrid, dimBlock >> > (d_arr, stride, flag);
            flag = 1;
        }

        bitonic_merge_in_block << < dimGrid, dimBlock >> > (d_arr, num_threads_in_block);
    }

    cudaMemcpy(arr, d_arr, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}
#endif

#ifdef CPU_RECURSIVE_VERSION

void BitonicSort(int, int, bool);
void BitonicMerge(int, int, bool);

void BitonicSort(int start, int n, bool isAscending) {
    if (n == 0) return;
    int m = n / 2;

    BitonicSort(start, m, true);
    BitonicSort(start + m, m, false);
    BitonicMerge(start, n, isAscending);
}

void BitonicMerge(int start, int n, bool isAscending) {
    if (n == 1) return;
    int m = n / 2;

    int temp;

    for (int i = start; i<start + m; i++) {
        if (isAscending == (arr[i] > arr[i + m])) {
            temp = arr[i];
            arr[i] = arr[i + m];
            arr[i + m] = temp;
        }
    }

    BitonicMerge(start, m, isAscending);
    BitonicMerge(start + m, m, isAscending);
}

#endif

void print() {
    for (int i = 0; i<arr_length; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

bool CheckPowOfTwo(int num) {
    int result = num & (num - 1);

    if (result == 0)
        return true;
    else
        return false;
}

void check() {
    bool result = true;

    for (int i = 0; i < arr_length; i++) {
        if (arr[i] != result_arr[i]) {
            result = false;
            printf("%d : your value is %d, result value is %d\n", i, arr[i], result_arr[i]);
        }
        //printf("%d : your value is %d, result value is %d\n", i, arr[i], result_arr[i]);
    }

    if (result) {
        printf("match\n");
    }
    else {
        printf("not match\n");
    }
}

int getPowNum(int n) {
    int count = 0;

    while (true) {
        n = n / 2;

        count++;
        if (n == 1)
            break;
    }

    return count;
}

int main(int argc, char * argv[]) {
    DS_timer timer(8);
    timer.initTimers();

    //arr_length = atoi(argv[1]);

    int padding = 0;
    int temp;
    int powNum = 0;
    int dx, dy;

    if (!CheckPowOfTwo(arr_length)) {
        padding = 1;
        temp = arr_length;

        while (true) {
            temp = temp >> 1;
            padding = padding << 1;
			if (temp == 0) break;
        }

        padding -= arr_length;
    }

    powNum = getPowNum(arr_length + padding);

    dx = (int)pow(2, powNum / 2);

    if (powNum % 2 == 0) {
        dy = (int)pow(2, powNum / 2);
    }

    else {
        dy = (int)pow(2, powNum / 2 + 1);
    }

    arr = new int[arr_length + padding];
    origin_arr = new int[arr_length + padding];
    result_arr = new int[arr_length + padding];

    for (int i = 0; i<arr_length; i++) {
        arr[i] = rand() % 10000 + 1;
        origin_arr[i] = arr[i];
    }

    for (int i = 0; i<padding; i++) {
        arr[arr_length + i] = INT_MAX;
        origin_arr[arr_length + i] = INT_MAX;
    }

    printf("array size = %d^%d\n", 2, powNum);
    
#ifdef CPU_RECURSIVE_VERSION
    timer.onTimer(0);
    BitonicSort(0, arr_length + padding, true);
    timer.offTimer(0);

    for (int i = 0; i < arr_length; i++) {
        result_arr[i] = arr[i];
    }

    check();

    for (int i = 0; i<arr_length; i++) {
        arr[i] = origin_arr[i];
    }

#endif

#ifdef CUDA_NORMAL_SEPERATE_MODE	
	for(int i=1 ; i<=3 ; i++) {
		timer.onTimer(i);

		bitonic_sort_from_host(arr_length + padding, dx, dy, 8 * pow(2, i-1));
		cudaThreadSynchronize();

		timer.offTimer(i);

		check();

		for (int i = 0; i < arr_length; i++) {
			arr[i] = origin_arr[i];
		}
	}
#endif

#ifdef CUDA_SHARED_MEMORY_MODE	
	for(int i=1 ; i<=3 ; i++) {
		timer.onTimer(i+3);

		bitonic_sort_from_host_shared(arr_length + padding, dx, dy, 8 * pow(2, i-1));
		cudaThreadSynchronize();

		timer.offTimer(i+3);

		check();

		for (int i = 0; i < arr_length; i++) {
			arr[i] = origin_arr[i];
		}
	}
#endif

    timer.setTimerName(0, "cpu recursive bitonic");
    timer.setTimerName(1, "gpu normal 8 thread in block ");
    timer.setTimerName(2, "gpu normal 16 thread in block ");
    timer.setTimerName(3, "gpu normal 32 thread in block ");
    timer.setTimerName(4, "gpu shared 8 thread in block ");
    timer.setTimerName(5, "gpu shared 16 thread in block ");
    timer.setTimerName(6, "gpu shared 32 thread in block ");

    free(arr);
    free(result_arr);
    free(origin_arr);

    timer.printTimer();

    return 0;
}
