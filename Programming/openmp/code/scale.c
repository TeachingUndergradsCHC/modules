/* 
 *  OpenMP implementatation of scaling a 2D array. This simple code is used to illustrat benefits of 
 *  multi-threaded parallelism and limits on performance scalability (i.e., Amdahl's Law)
 *
 *  @author Apan Qasem 
 */
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#include <omp.h>

#define REPS 100

double t0;
double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char *argv[]) {
  int **a, **b;
  
  int M = atoi(argv[1]);
  int N = atoi(argv[2]);

  omp_set_num_threads(N);

  a = (int **) malloc(sizeof(int *) * M);
  b = (int **) malloc(sizeof(int *) * M);
  
  int i, j, k;
  for (i = 0; i < M; i++) {
    a[i] = (int *) malloc(sizeof(int) * M);
    b[i] = (int *) malloc(sizeof(int) * M);
  }

  for (j = 0; j < M; j++)
    for (i = 0; i < M; i++)
      b[i][j] = i + j;

  t0 = mysecond();
#pragma omp parallel for private(j,i)
  for (k = 0; k < REPS; k++) {
    for (j = 0; j < M; j++) 
      for (i = 0; i < M; i++)
	a[i][j] = b[i][j] * 17;
  }

  t0 = (mysecond() - t0) * 1.e3;
  printf("parallel loop = %3.2f ms\n", t0);

  return 0;

}
