/* 
 * Recursive parallel-hybrid implementation of MergeSort (not optimized!)
 * This code is to be used as examples in module [B1] Hybrid Algorithm
 *
 * parts of the code borrow from
 *   : https://stackoverflow.com/questions/13811114/parallel-merge-sort-in-openmp
 * 
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/04/21 
 * 
 * @update: 07/18/22 
 */


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#define VAL_RANGE 1024
#define ELEMENTS_TO_VERIFY 5
#define PAR_THRESHOLD 100


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

/* 
 * display array contents 
 */
void display(double values[], long long N) {
  for (int i = 0; i < N; i++)
    fprintf(stdout, "%3.4f ", values[i]);
  fprintf(stdout, "\n");
}


#pragma omp declare target 
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

   while (i < n/2) { 
      aux[aux_index] = values[i];
      aux_index++;
      i++;
   }
   while (j < n) { 
      aux[aux_index] = values[j];
      aux_index++;
      j++;
   }
   memcpy(values, aux, n * sizeof(double));
} 
#pragma omp end declare target 

#pragma omp declare target 
void merge_sort_gpu(double *values, int n, double *aux) {
   if (n < 2) return;

   unsigned int ub = n/2;
   merge_sort_gpu(values, n/2, aux);
   merge_sort_gpu(values + (n/2), n - (n/2), aux + n/2);
   merge(values, n, aux);
}
#pragma omp end declare target 

void merge_sort_cpu(double *values, int n, double *aux) {
   if (n < 2)
     return;

   #pragma omp task shared(values) if (n > PAR_THRESHOLD)
   merge_sort_cpu(values, n/2, aux);

   #pragma omp task shared(values) if (n > PAR_THRESHOLD)
   merge_sort_cpu(values + (n/2), n - (n/2), aux + n/2);

   #pragma omp taskwait
   merge(values, n, aux);
}

void merge_sort_driver(double *values, int n, double *aux) {
   if (n < 2)
     return;

   unsigned int ub = n/2;
   #pragma omp target map(tofrom:values[0:ub]) map(tofrom:aux[0:ub])
   {
     merge_sort_gpu(values, n/2, aux);
   }
   #pragma omp task shared(values) if (n > PAR_THRESHOLD)
   merge_sort_cpu(values + (n/2), n - (n/2), aux + n/2);

   #pragma omp taskwait
   merge(values, n, aux);
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
  
  omp_set_num_threads(threads);
  
  double *values = (double *) malloc(sizeof(double) * N);
  double *aux = (double *) malloc(sizeof(double) * N);

  for (int i = 0; i < N; i++) 
    values[i] = rand() / (double) (RAND_MAX/VAL_RANGE);


  omp_set_dynamic(0);   

  double t0 = timer();
  #pragma omp parallel
  {
    #pragma omp single
    merge_sort_driver(values, N, aux);
  }   
  t0 = (timer() - t0) * 1.e3;
  
  fprintf(stdout, "Sorted values [0..%d]: ", ELEMENTS_TO_VERIFY - 1);
  display(values, ELEMENTS_TO_VERIFY);
  fprintf(stdout, "Execution time = %3.2f ms\n", t0); 
  
  
  return 0; 
}
