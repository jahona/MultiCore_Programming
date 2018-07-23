#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "DS_timer.h"

const int MAX = 1024*1024*1024;

float * arr = new float[MAX];

void inputArr() {
	srand(time(NULL));
    for(int i=0 ; i<MAX ; i++) {
        //arr[i] = 10*((float)rand()/RAND_MAX);
		arr[i] = (rand() %100) /10;
    }
}

void printBin(int version, int * bin) {
	printf("%d version start\n", version);

	for(int i=0 ; i<10 ; i++) {
		printf("[%d] => %d\n", i, bin[i]);
	}
}

void serial() {
	int bin[10] = { 0, };

	for (int i = 0; i<MAX; i++) {
		bin[(int)arr[i]] += 1;
	}

	printBin(0, bin);
}

void version1() {
    int bin[10] = {0, };

	#pragma omp parallel for num_threads(4)
    for(int i=0 ; i<MAX ; i++) {
        #pragma omp atomic
        bin[(int)arr[i]]+=1;
    }

	printBin(1, bin);
}

void version2() {
    int localBin[10] = {0, };
    int bin[10] = {0, };

    #pragma omp parallel num_threads(4) firstprivate(localBin)
    {
        #pragma omp for
        for(int i=0 ; i<MAX ; i++) {
            localBin[(int)arr[i]]++;
        }

        for(int j=0 ; j<10 ; j++) {
            #pragma omp atomic
            bin[j] += localBin[j];
        }        
    }

	printBin(2, bin);
}

void version3() {
    int * localBin[4];
    int sum[2][10] = {0, };
	int bin[10] = {0, };

    int pid;

    for(int i=0 ; i<4 ; i++) {
        localBin[i] = new int[10];
		for(int j=0 ; j<10 ; j++) 
			localBin[i][j] = 0;
	}

	#pragma omp parallel num_threads(4) private(pid)
	{
        pid = omp_get_thread_num();

        #pragma omp for
        for(int i=0 ; i<MAX ; i++) {
            localBin[pid][(int)arr[i]]++;
		}
    }

	#pragma omp parallel num_threads(2) private(pid)
    {
        pid = omp_get_thread_num();

        if(pid%2==0) {
            for(int i=0 ; i<10 ; i++) {
                sum[0][i] += localBin[0][i] + localBin[2][i];
            }
        }

        else{
            for(int i=0 ; i<10 ; i++) {
                sum[1][i] += localBin[1][i] + localBin[3][i];
			}
        }
    }

    for(int i=0 ; i<10 ; i++) {
        bin[i] = sum[0][i] + sum[1][i];
    }

	printBin(3, bin);
}

int main() {
    inputArr();

    DS_timer timer(4);
    timer.initTimers();

	timer.onTimer(0);
	serial();
	timer.offTimer(0);

    timer.onTimer(1);
    version1();
    timer.offTimer(1);

    timer.onTimer(2);
    version2();
    timer.offTimer(2);

    timer.onTimer(3);
    version3();
    timer.offTimer(3);

    timer.printTimer();

	getchar();
    return 0;
}