/* 
 * Code to simulate a "little"-core workload for lab assignment in [A2] Task Mapping on Soft Heterogeneous Systems. 
 * Workload consists of a parallel implementation of matrix-vector multiplication (not optimized).
 * Implementation is not optimized! Only meant to be used in conjunction with lab assignment. 
 *
 * Clock frequency has low impact on the performance of this workload. In a task mapping scenario, 
 * this workload should be mapped to the "little" cores to minimize performance loss. 
 *
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/02/20 
 * 
 * @update: 03/09/21
 */

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<unistd.h>
#include<omp.h>

#define VAL_RANGE 1024

/* timer function */ 
double get_time_in_seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

long dot_prod(long *x, long *y, long n) {
  long sum = 0;
  long i;
#pragma omp parallel for reduction(+:sum)
  for (i = 0; i < n; i++)
      sum += x[i] * y[i];
  return sum;
}

void matrix_vector_mult(long **mat, long *vec, long *result, 
			long long rows, long long cols) { 

  /* not parallelelized to ensure runtimes are more meaningful */
  long i;
  for (i = 0; i < rows; i++)
    result[i] = dot_prod(mat[i], vec, cols);
}

void display_matrix(const long **matrix, long long N) {
  long i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) 
      printf("%lu ", matrix[i][j]);
    printf("\n");
  }
}

int main(int argc, char *argv[]) {

  if (argc < 4) {
    printf("usage: \n");
    printf("       ./matvec N n t\n");
    printf("       N = matrix dimension\n");
    printf("       n = number of reps\n");
    printf("       t = number of threads\n");
    exit(0);
  }

  /* matrix dimenstion, assume N x N matrix and vector of size of N*/ 
  long long N = atoi(argv[1]);

  /* number of reps;  control running time of program */
  unsigned n = atoi(argv[2]);

  /* number of OpenMP threads */
  unsigned threads = atoi(argv[3]);
  omp_set_num_threads(threads);

  long **matrix;
  long *vec;
  long *result;

  long i, j;
  double start_time, end_time;


  /* memory allocation and initilization */
#ifdef INIT_TIMER
  start_time = get_time_in_seconds();
#endif
  matrix = (long **) malloc(sizeof(long *) * N);
  for (i = 0; i < N; i++)
    matrix[i] = (long *) malloc(sizeof(long) * N);

  vec = (long *) malloc(sizeof(long) * N);
  result = (long *) malloc(sizeof(long) * N);

  for (i = 0; i < N; i++) 
    for (j = 0; j < N; j++) 
      matrix[i][j] = rand() / (long) (RAND_MAX/VAL_RANGE);
     
  for (i = 0; i < N; i++)
    vec[i] = rand() / (long) (RAND_MAX/VAL_RANGE);
#ifdef INIT_TIMER
  end_time = get_time_in_seconds();
  fprintf(stdout, "Initialization time = %.3f s\n", end_time - start_time);
#endif
  
  /* computation */ 
  start_time = get_time_in_seconds();
  for (i = 0; i < n; i++)
    matrix_vector_mult(matrix, vec, result, N, N);
  sleep(3);
  end_time = get_time_in_seconds();

  /* verification (by inspection only) */
#ifdef VERIFY
  fprintf(stdout, "Verification: "); 
  for (unsigned i = 0; i < 1; i++) 
    fprintf(stdout, "result[%d] = %lu\n", i, result[i]);
#endif
  
  fprintf(stdout, "\033[1;35m[wk1] compute time = %.3f s\n\033[0m", end_time - start_time);

  return 0;
}
