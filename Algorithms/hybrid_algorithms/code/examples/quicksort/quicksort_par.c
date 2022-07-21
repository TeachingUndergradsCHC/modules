/* 
 * Recursive parallel implementation of Quicksort (not optimized!)
 * This code is to be used in conjunction with exercises in module [B1] Hybrid Algorithm
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

void swap(double *x, double *y) {
  double tmp;
  tmp = (*x);
  (*x) = (*y);
  (*y) = tmp;
  return;
}
/*
 * partition array for quicksort
 *     - move pivot to far right
 *     - accumulate values smaller than pivot to the left
 */
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

/* 
 * recursive quicksort 
 */ 
void quickSort(double values[], int left, int right) {
  #pragma omp parallel
  {
    #pragma omp single
    {
      if (left < right) {
	int pivotIndex = (left + right)/2;
	int pivotNewIndex = partition(values, left, right, pivotIndex);

        #pragma omp task
	quickSort(values, left, pivotNewIndex - 1);
        #pragma omp task
	quickSort(values, pivotNewIndex + 1, right);
      }
    }
  }
  return;
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
  quickSort(values, 0, N - 1);
  t0 = (timer() - t0) * 1.e3;

  fprintf(stdout, "Sorted values [0..%d]: ", ELEMENTS_TO_VERIFY - 1);
  display(values, ELEMENTS_TO_VERIFY);
  fprintf(stdout, "Execution time = %3.2f ms\n", t0); 
  
  return 0;
}

 
