
/* 
 * Code to simulate a "big"-core workload for lab assignment in [A2] Task Mapping on Soft Heterogeneous Systems. 
 * Workload consists of a parallel implementation of quicksort.
 * Implementation not optimized! Only meant to be used in conjunction with lab assignment. 

 * The performance of the workload increases proportianlly with clock frequency. In a task mapping scenario, 
 * this workload should be mapped to the "big" cores for optimal performance. 
 *
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/02/20 
 * 
 * @update: 03/12/21
 */

#include<stdlib.h>
#include<stdio.h>
#include<sys/time.h>
#include<omp.h>


const int VAL_RANGE = 1024;
const unsigned ELEMENTS_TO_VERIFY = 1;

/* timer function */ 
double get_time_in_seconds() {
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
  int pivotValue = values[pivotIndex];
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
  if (left < right) {
    int pivotIndex = (left + right)/2;
    
    int pivotNewIndex = partition(values, left, right, pivotIndex);
#pragma omp task shared(values) firstprivate(left, pivotNewIndex)
    quickSort(values, left, pivotNewIndex - 1);
#pragma omp task shared(values) firstprivate(right, pivotNewIndex)
    quickSort(values, pivotNewIndex + 1, right);
#pragma omp taskwait
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

  if (argc < 2) {
    printf("usage: \n");
    printf("       ./big_wkld N t\n");
    printf("       N = input size\n"); 
    printf("       t = number of OpenMP threads\n"); 
    exit(0);
  }
  
  long long N = atoi(argv[1]);

  unsigned threads = atoi(argv[2]);
  omp_set_num_threads(threads);

  double start_time, end_time;
  
  double *values = (double *) malloc(sizeof(double) * N);
  for (int i = 0; i < N; i++) 
    values[i] = rand() / (double) (RAND_MAX/VAL_RANGE);
  
  /* computation */ 
  start_time = get_time_in_seconds();
  quickSort(values, 0, N - 1);
  end_time = get_time_in_seconds();
  fprintf(stdout, "\033[1;36m[wk0] compute time = %.3f s\n\033[0m", end_time - start_time);

#ifdef VERIFY
  fprintf(stdout, "Verification: ");
  display(values, ELEMENTS_TO_VERIFY);
#endif
  
  return 0;
}

 
