/* 
 * Canonical recursive implementation of MergeSort (not optimized!)
 * This code is to be used as examples in module [B1] Hybrid Algorithm
 *
 * 
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/04/21 
 * 
 * @update: 07/18/22 
 */


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define VAL_RANGE 1024
#define ELEMENTS_TO_VERIFY 5


/*
 *  retrieve time in seconds from getitimeofday()
 */
double timer() {
  struct timeval tp;
  struct timezone tzp;
  int i;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


void merge(double *values, int n, double *aux) {
   int i = 0;
   int j = n/2;
   int aux_index = 0;

   while (i < n/2 && j < n) {
     if (values[i] < values[j]) {
       aux[aux_index] = values[i];
       aux_index++; i++;
     } else {
       aux[aux_index] = values[j];
       aux_index++; j++;
     }
   }

   while (i < n/2) { /* finish up lower half */
      aux[aux_index] = values[i];
      aux_index++;
      i++;
   }
   while (j < n) { /* finish up upper half */
      aux[aux_index] = values[j];
      aux_index++;
      j++;
   }
   memcpy(values, aux, n * sizeof(double));
} 

void merge_sort(double *values, int n, double *aux) {
   if (n < 2) return;

   #pragma omp task shared(values)
   //   if (n > TASK_SIZE)
   merge_sort(values, n/2, aux);

   #pragma omp task shared(values)
   //   if (n > TASK_SIZE)
   merge_sort(values + (n/2), n - (n/2), aux + n/2);

   #pragma omp taskwait
   merge(values, n, aux);
}

/* 
 * display array contents 
 */
void display(double values[], long long N) {
  for (int i = 0; i < N; i++)
    fprintf(stdout, "%3.4f ", values[i]);
  fprintf(stdout, "\n");
}

int main(int argc, char *argv[]) {

  if (argc < 3) {
    printf("usage: \n");
    printf("       ./mergesort N threads\n");
    printf("       N = input size\n"); 
    printf("       t = number of OpenMP threads\n"); 
    exit(0);
  }
  
  long long N = atoi(argv[1]);
  unsigned threads = atoi(argv[2]);
  
  double *values = (double *) malloc(sizeof(double) * N);
  double *aux = (double *) malloc(sizeof(double) * N);

  for (int i = 0; i < N; i++) 
    values[i] = rand() / (double) (RAND_MAX/VAL_RANGE);
  
  double t0 = timer();
  merge_sort(values, N, aux);
  t0 = (timer() - t0) * 1.e3;

  
  fprintf(stdout, "Sorted values [0..%d]: ", ELEMENTS_TO_VERIFY - 1);
  display(values, ELEMENTS_TO_VERIFY);
  fprintf(stdout, "Execution time = %3.2f ms\n", t0); 
  
  
  return 0; 
}
