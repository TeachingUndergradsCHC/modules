/* 
 * Code to simulate a cache-intensive workload for lab assignment in [A2] Task Mapping on Soft Heterogeneous Systems. 
 * Workload consists of a simple parallel initilization routine. 
 * Implementation is not optimized! Only meant to be used in conjunction with lab assignment. 

 * The performance of the workload is sensitive to the cache siz. In a task mapping scenario, 
 * this workload should be mapped to the set of cores with largest cache. 
 *
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/02/20 
 * 
 * @update: 03/12/21
 */
#include<stdlib.h>
#include<stdio.h>
#include<sys/time.h>
#include <omp.h>

#define ELEMENTS_TO_VERIFY 1

/* timer function */ 
double get_time_in_seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char *argv[]) {
  int **a;
  
  if (argc < 2) {
    printf("usage: \n");
    printf("       ./cache_wkld N t\n");
    printf("       N = input size\n"); 
    printf("       t = number of OpenMP threads\n"); 
    exit(0);
  }
  
  long long N = atoi(argv[1]);
  unsigned threads = atoi(argv[2]);
  omp_set_num_threads(threads);

  a = (int **) malloc(sizeof(int *) * N);
  int i,j,k;
  for (i = 0; i < N; i++) 
    a[i] = (int *) malloc(sizeof(int) * N);

  double start_time, end_time;

  start_time = get_time_in_seconds();
#pragma omp parallel for private(j,i) 
  for (k = 0; k < 100; k++) {
    for (j = 1; j < N; j++) 
      for (i = 1; i < N; i++)
	a[i][j] = 17;
  }
  end_time = get_time_in_seconds();
  fprintf(stdout, "\n\033[0;33mCompute time = %.3f s\n\033[0m", end_time - start_time);

  fprintf(stdout, "Verification: ");
  for (int i = 0; i < ELEMENTS_TO_VERIFY; i++) 
    fprintf(stdout, "%d\n", a[0][i]);
  return 0;
}
