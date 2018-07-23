#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char * argv[]) {
	if(argc == 1) {
		printf("error");
		exit(1);
	}

	int n = atoi(argv[1]);

#pragma omp parallel num_threads(n)
	printf("Hello OpenMP\n");

	return 0;
}