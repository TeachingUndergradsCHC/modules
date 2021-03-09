/* 
 * OpenMP implementation of matrix-vector multiplication (not optimized).
 * To be used with the in-class demo in model [A2]: Task Mapping on Soft Heterogeneous Systems
 *  
 * Apan Qasem <apan@txtstate.edu>
 * last updated: 03/09/2021
 */
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<omp.h>

#define VAL_RANGE 1023

/* timer function */ 
double get_time_in_seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

double dot_prod(double *x, double *y, int n) {
  double sum = 0.0;
  int i;
#pragma omp parallel for reduction(+:sum)
  for (i = 0; i < n; i++)
      sum += x[i] * y[i];
  return sum;
}

void matrix_vector_mult(double **mat, double *vec, double *result, 
			long long rows, long long cols) { 

  /* not parallelelized to ensure runtimes are more meaningful */
  int i;
  for (i = 0; i < rows; i++)
    result[i] = dot_prod(mat[i], vec, cols);
}

void display_matrix(const double **matrix, long long N) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) 
      printf("%3.4f ", matrix[i][j]);
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

  double **matrix;
  double *vec;
  double *result;

  int i, j;
  double start_time, end_time;


  /* memory allocation and initilization */
#ifdef INIT_TIMER
  start_time = get_time_in_seconds();
#endif
  matrix = (double **) malloc(sizeof(double *) * N);
  for (i = 0; i < N; i++)
    matrix[i] = (double *) malloc(sizeof(double) * N);

  vec = (double *) malloc(sizeof(double) * N);
  result = (double *) malloc(sizeof(double) * N);

  for (i = 0; i < N; i++) 
    for (j = 0; j < N; j++) 
      matrix[i][j] = rand() / (double) (RAND_MAX/VAL_RANGE);
     
  for (i = 0; i < N; i++)
    vec[i] = rand() / (double) (RAND_MAX/VAL_RANGE);
#ifdef INIT_TIMER
  end_time = get_time_in_seconds();
  fprintf(stdout, "Initialization time = %.3f s\n", end_time - start_time);
#endif
  
  /* computation */ 
  start_time = get_time_in_seconds();
  for (i = 0; i < n; i++)
    matrix_vector_mult(matrix, vec, result, N, N);
  end_time = get_time_in_seconds();

  /* verification (by inspection only) */
  fprintf(stdout, "Verification: "); 
  for (unsigned i = 0; i < 1; i++) 
    fprintf(stdout, "result[%d] = %3.2e\n", i, result[i]);

  fprintf(stdout, "\n\033[0;33mCompute time = %.3f s\n\033[0m", end_time - start_time);

  return 0;
}
