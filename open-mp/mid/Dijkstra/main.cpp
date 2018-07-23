#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "DS_timer.h"	// for timer
#include <omp.h>
#include <math.h>

#define MAX 10000
#define NUM_THREADS 4

using namespace std;

int N;
int ** map;
bool * visit;
int start_point, end_point;
unsigned int Min = 0xFFFFFFFF;

/*
Array에 랜덤한 값을 넣기
*/

double logB(double x, double base) { 
	return log(x)/log(base);
}

void input_random_toArray(int ** arr, int n) {
	for(int i=0; i<n; i++){
		for(int j=0; j<n;j++){
			if(i==j){
				arr[i][j]=0;
			} else {
				arr[i][j]=rand()%80+20;
			}
		}
	}
}

/*
이차원의 배열을 동적 할당하여 생성 및 반환
*/
int** create_array_dynamic(int n) {
   int** arr;
   int temp;
   int cut_line;

   arr = new int*[n];

   for(int i = 0; i < n; i++ ) {
      arr[i] = new int[n];
   }
   
   for(int i=0; i<n; i++){
      for(int j=0; j<n;j++){
         arr[i][j]= MAX;
         if(i==j){
            arr[i][j]=0;
         }
      }
   }
   /*
   cut_line = n/1000;
   for(int i=0; i<n; i++){
      for(int j=0; j<cut_line;j++){
         temp = rand()%n;
         if( temp != i ) {
            arr[i][temp]= rand() % 100;
         }
         else j--;
      }
   }
   
   int not_continue;
   if( n != 1000 ) not_continue = n/1000+n%1000;
   else  not_continue = 30;
   */
   int not_continue = logB(n,2);
   printf("not _ continue : %d\n",not_continue);

   for(int i=0; i<n; i++){
      for(int j=0; j<not_continue;j++){
		 temp = rand()%n;
         if( temp != i ) {
            arr[i][temp]= rand() % 20+80;
         }

	  }
   }


   return arr;
}

/*
각 실행을 위해 N을 받고 2차원 배열을 동적 할당
start 지점과 end 지점을 입력받음.
*/
void setting_running_env() {
	printf("N = ");
	scanf("%d", &N);
	map = create_array_dynamic(N);
	//input_random_toArray(map, N);
	printf("start : ");
	scanf("%d", &start_point);
	printf("end : ");
	scanf("%d", &end_point);
	printf("\n\n\n");
}

/*
배열을 출력
*/
void Array_print(int n) {
	for(int i=0 ; i<n ; i++) {
		for(int j=0 ; j<n ; j++) {
			printf("%d ", map[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*
다익스트라 알고리즘을 이용한 최단 거리 알고리즘
*/
int dijkstra_serial(int **arr, int n, int start, int end) {
	FILE * fp = fopen("dijkstra_serial.csv", "w+");

	int * value = new int[n];
	int * f = new int[n];
	int * index = new int[n];
	int * intimacy_value = new int[n];
	int * intimacy_num = new int[n];
	int * intimacy_sum = new int[n];

	int min, point;

	for (int i = 0; i<n; i++) {
		value[i] = MAX;
		f[i] = 0;
		index[i] = 0;
		intimacy_num[i] = 1;
		intimacy_sum[i] = 0;
	}

	//start vertex
	value[start] = 0;

	for (int i = 0; i<n; i++)
	{
		min = MAX;
		for (int j = 0; j<n; j++) {
			if ((value[j]<min) && f[j] == 0) {
				min = value[j];
				point = j;
			}
		}
		f[point] = 1;
		for (int j = 0; j<n; j++) {
			if (value[point] + arr[point][j] < value[j]) {
				value[j] = value[point] + arr[point][j];
				index[j] = point;
			}
		}
	}

	fprintf(fp, "거리, 나, 친구, 경로, 친밀도\n");
	for(int i=0 ; i<n ; i++) {
		if(i!=start) {
			fprintf(fp, "%d, %d, %d, %d ", value[i], start, i, i);

			point = i;

			do {
				intimacy_num[i]++;
				point = index[point];
			 } while (index[point]);

			 if (intimacy_num[i] == 2 && value[i] != MAX) {
				intimacy_value[i] = value[i];
			 }
			 else if(intimacy_num[i] == 2 && value[i] == MAX) {
				 intimacy_value[i]  = 0;
			 }
			 else {
				point = i;
				do {

				   intimacy_sum[i] += arr[index[point]][point] * intimacy_num[i];
				   intimacy_num[i]--;
				   fprintf(fp, "<- %d", index[point]);
				   point = index[point];
				} while (index[point]);

				intimacy_value[i] = (float)value[i] / intimacy_sum[i] * 100;
			 }

			fprintf(fp, ", %d\n", intimacy_value[i]);
		}
	}

	int result = value[end_point];

	delete[] (value);
	delete[] (f);
	delete[] (index);
	delete[] (intimacy_value);
	delete[] (intimacy_num);
	delete[] (intimacy_sum);
	
	return result;
}

/*
다익스트라 알고리즘에 병렬화를 시킨 코드
*/
int dijkstra_parallel(int** arr, int n, int start, int end, int thread_count) {
	FILE * fp = fopen("dijkstra_parallel.csv", "w+");

	int min[NUM_THREADS];
	int points[NUM_THREADS];
	int point;
	int * value = new int[n];
	int * f = new int[n];
	int * index = new int[n];

	int * intimacy_value = new int[n];
	int * intimacy_num = new int[n];
	int * intimacy_sum = new int[n];
	
	#pragma omp parallel for
	for (int i = 0; i<n; i++) {
		value[i] = MAX;
		f[i] = 0;
		index[i] = 0;
		intimacy_num[i] = 1;
		intimacy_sum[i] = 0;
	}

	// start vertex
	value[start] = 0;

	for (int i = 0; i<n; i++) {
		// min 초기화
		#pragma omp parallel for num_threads(thread_count)
		for (int j = 0; j<NUM_THREADS; j++)
			min[j] = MAX;

		// 가장 가까운 정점 찾기
		#pragma omp parallel num_threads(thread_count)
		{
			int tID = omp_get_thread_num();

			// 각 스레드에게 각자 영역에서 min 값을 구함
			#pragma omp for
			for (int j = 0; j<n; j++) {
				if ((value[j]<min[tID]) && f[j] == 0) {
					min[tID] = value[j];
					points[tID] = j;
				}
			}

			// 전체 스레드에서 min 값을 구함
			int offset = 1;
			while (offset < NUM_THREADS) {
				if (tID % (2 * offset) == 0) {
					if (min[tID] > min[tID + offset]) {
						min[tID] = min[tID + offset];
						points[tID] = points[tID + offset];
					}
				}

				#pragma omp barrier

				offset *= 2;
			}
		}

		point = points[0];

		f[point] = 1;

		// get the most shortest distance from each core
		for (int j = 0; j<n; j++) {
			if (value[point] + arr[point][j] < value[j]) {
				value[j] = value[point] + arr[point][j];
				index[j] = point;
			}
		}
	}
	
	fprintf(fp, "거리, 나, 친구, 경로, 친밀도\n");
	for(int i=0 ; i<n ; i++) {
		if(i!=start) {
			fprintf(fp, "%d, %d, %d, %d ", value[i], start, i, i);

			point = i;

			do {
				intimacy_num[i]++;
				point = index[point];
			 } while (index[point]);

			 if (intimacy_num[i] == 2 && value[i] != MAX) {
				intimacy_value[i] = value[i];
			 }
			 else if(intimacy_num[i] == 2 && value[i] == MAX) {
				 intimacy_value[i]  = 0;
			 }
			 else {
				point = i;
				do {

				   intimacy_sum[i] += arr[index[point]][point] * intimacy_num[i];
				   intimacy_num[i]--;
				   fprintf(fp, "<- %d", index[point]);
				   point = index[point];
				} while (index[point]);

				intimacy_value[i] = (float)value[i] / intimacy_sum[i] * 100;
			 }

			fprintf(fp, ", %d\n", intimacy_value[i]);
		}
	}
	
	int result = value[end_point];

	delete[] (value);
	delete[] (f);
	delete[] (index);
	delete[] (intimacy_value);
	delete[] (intimacy_num);
	delete[] (intimacy_sum);

	return result;
}

int main() {
	char ch;

	DS_timer timer(8);
	timer.initTimers();

	printf("dijkstra serial VS dijkstra parallel\n\n");
	setting_running_env();

	//Array_print(N);

	// dijkstra serial
	printf("1. dijkstra serial\n");
	timer.onTimer(1);
	Min = dijkstra_serial(map, N, start_point, end_point);
	timer.offTimer(1);
	printf("최단거리 : %u\n", Min);

	// dijkstra parallel
	printf("2. dijkstra parallel. thread = 1\n");
	timer.onTimer(2);
	Min = dijkstra_parallel(map, N, start_point, end_point, 1);
	timer.offTimer(2);
	printf("최단거리 : %u\n", Min);

	printf("3. dijkstra parallel. thread = 2\n");
	timer.onTimer(3);
	Min = dijkstra_parallel(map, N, start_point, end_point, 2);
	timer.offTimer(3);
	printf("최단거리 : %u\n", Min);

	printf("4. dijkstra parallel. thread = 4\n");
	timer.onTimer(4);
	Min = dijkstra_parallel(map, N, start_point, end_point, 4);
	timer.offTimer(4);
	printf("최단거리 : %u\n", Min);

	timer.printTimer();

	return 0;
}