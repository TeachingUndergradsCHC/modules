#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#include <omp.h>

#define REPS 10000

double t0;

double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char *argv[]) {
  
  int M = atoi(argv[1]);  // size of vectors 
  int N = atoi(argv[2]);  // number of OpenMP threads

  float*a, *b;
  a = (float*) malloc(sizeof(float) * M);
  b = (float*) malloc(sizeof(float) * M);
  
  int i, j, k;
  for (i = 0; i < M; i++) {
    a[i] = i; 
    b[i] = i + 3; 
  }

  omp_set_num_threads(N);

  float sum = 0;
  t0 = mysecond();
  for (j = 0; j < REPS; j++) {
#pragma omp parallel for reduction(+:sum)
    for (i = k; i < M; i++)
	sum += a[i] * b[i];
    }
  t0 = (mysecond() - t0) * 1.e3;

  fprintf(stdout, "result = %1.3e\n", sum);
  fprintf(stdout, "parallel loop = %3.2f ms\n", t0);

  return 0;

}
