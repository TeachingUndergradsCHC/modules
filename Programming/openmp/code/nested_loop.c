/* 
 *  This example illustrates nested loop parallelization with OpenMP
 *
 *  @author Apan Qasem 
 */
#include<stdlib.h>
#include<stdio.h>
#include <omp.h>

#define M 4

int main(int argc, char *argv[]) {
  
  int N = atoi(argv[1]);
  omp_set_num_threads(N);

  int j, k;
#pragma omp parallel for private(j) collapse(2)
  for (k = 0; k < M; k++)
    for (j = 0; j < M; j++)
      printf("I am thread %d in iteration (%d,%d)\n", omp_get_thread_num(), k,j);
  return 0;
}
