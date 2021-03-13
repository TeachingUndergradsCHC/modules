/* 
 * Recursive implementation of Quicksort (not optimized!)
 * This code is to be used in conjunction with exercises in module [B1] Hybrid Algorithm
 *
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/02/20 
 * 
 * @update: 03/13/21
 */

#include<stdlib.h>
#include<stdio.h>

#define VAL_RANGE 1024
#define ELEMENTS_TO_VERIFY 5

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
  if (left < right) {
    int pivotIndex = (left + right)/2;
    int pivotNewIndex = partition(values, left, right, pivotIndex);

    quickSort(values, left, pivotNewIndex - 1);
    quickSort(values, pivotNewIndex + 1, right);
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
    printf("       ./quicksort N\n");
    printf("       N = input size\n"); 
    exit(0);
  }
  
  long long N = atoi(argv[1]);

  double *values = (double *) malloc(sizeof(double) * N);
  for (int i = 0; i < N; i++) 
    values[i] = rand() / (double) (RAND_MAX/VAL_RANGE);

  quickSort(values, 0, N - 1);

  fprintf(stdout, "Sorted values [0..%d]: ", ELEMENTS_TO_VERIFY - 1);
  display(values, ELEMENTS_TO_VERIFY);
  
  return 0;
}

 
