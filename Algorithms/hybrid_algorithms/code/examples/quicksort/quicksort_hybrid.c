/* 
 * Recursive hybrid-parallel implementation of Quicksort (not optimized!)
 * This code is to be used in conjunction with exercises in module [B1] Hybrid Algorithm
 *
 * Need OpenMP 5.0 and GCC 11 to compiler (recursive offloading)
 *
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/02/20 
 * 
 * @update: 03/13/21
 */

#include<stdlib.h>
#include<stdio.h>
#include<omp.h>
#include <sys/time.h>

#define VAL_RANGE 1024
#define ELEMENTS_TO_VERIFY 10

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
void swap(double *x, double *y) {
  double tmp;
  tmp = (*x);
  (*x) = (*y);
  (*y) = tmp;
  return;
}
#pragma omp end declare target 

/*
 * partition array for quicksort
 *     - move pivot to far right
 *     - accumulate values smaller than pivot to the left
 */
#pragma omp declare target 
int partition(double values[], int left, int right, int pivotIndex) {
  double pivotValue = values[pivotIndex];
  swap(&values[pivotIndex],&values[right]);  // Move pivot to end
  int storeIndex = left;
  for(int i = left; i < right; i++) {
    if (values[i] < pivotValue) {
      swap(&values[i],&values[storeIndex]);
      storeIndex++;
    }
  }
  swap(&values[storeIndex],&values[right]);  // Move pivot to its final place
  return storeIndex;
}
#pragma omp end declare target 

#pragma omp declare target 
void quickSort_gpu(double values[], int left, int right) {
  if (left < right) {
    int pivotIndex = (left + right)/2;
    int pivotNewIndex = partition(values, left, right, pivotIndex);
    
    quickSort_gpu(values, left, pivotNewIndex - 1);
    quickSort_gpu(values, pivotNewIndex + 1, right);
  }
  return;
}
#pragma omp end declare target 

void quickSort_cpu(double values[], int left, int right) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      if (left < right) {
	int pivotIndex = (left + right)/2;
	int pivotNewIndex = partition(values, left, right, pivotIndex);

        #pragma omp task
	quickSort_cpu(values, left, pivotNewIndex - 1);
        #pragma omp task
	quickSort_cpu(values, pivotNewIndex + 1, right);
      }
    }
  }
  return;
}

void quickSort_driver(double values[], int left, int right) {
      if (left < right) {
	int pivotIndex = (left + right)/2;
	int pivotNewIndex = partition(values, left, right, pivotIndex);
	
	#pragma omp target map(tofrom:values[left:pivotNewIndex])
	{
	  quickSort_gpu(values, left, pivotNewIndex - 1);
        }
        #pragma omp task
	  quickSort_cpu(values, pivotNewIndex + 1, right);
      }
  return;
}


int main(int argc, char *argv[]) {

  if (argc < 3) {
    printf("usage: \n");
    printf("       ./quicksort N threads\n");
    printf("       N = input size\n"); 
    printf("       t = number of OpenMP threads\n"); 
    exit(0);
  }
  
  long long N = atoi(argv[1]);
  unsigned threads = atoi(argv[2]);

  omp_set_num_threads(threads);
  
  double *values = (double *) malloc(sizeof(double) * N);
  for (int i = 0; i < N; i++) 
    values[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

  double t0 = timer();
  quickSort_driver(values, 0, N - 1);
  t0 = (timer() - t0) * 1.e3;

  fprintf(stdout, "Sorted values [0..%d]: ", ELEMENTS_TO_VERIFY - 1);
  display(values, ELEMENTS_TO_VERIFY);
  fprintf(stdout, "Execution time = %3.2f ms\n", t0); 
  
  return 0;
}

 
