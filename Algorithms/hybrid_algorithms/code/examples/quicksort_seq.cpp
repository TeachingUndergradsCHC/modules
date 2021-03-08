#include<iostream>
#include<cstdlib>
#include<omp.h>

using namespace std; 

const int N = 500000;
const int RAND_RANGE_POPULATION = 20;
const int RAND_RANGE_SEARCH = 100;


int partition(int values[], int left, int right, int pivotIndex) {
  int pivotValue = values[pivotIndex];
  swap(values[pivotIndex],values[right]);  // Move pivot to end
  int storeIndex = left;
  for(int i = left; i < right; i++) {
    if (values[i] < pivotValue) {
      swap(values[i],values[storeIndex]);
      storeIndex++;
    }
  }
  swap(values[storeIndex],values[right]);  // Move pivot to its final place
  return storeIndex;
}


void quickSort(int values[], int left, int right) {
  if (left < right) {
    int pivotIndex = (left + right)/2;
    
    int pivotNewIndex = partition(values, left, right, pivotIndex);
    quickSort(values, left, pivotNewIndex - 1);
    quickSort(values, pivotNewIndex + 1, right);
  }
  return;
}

int main() {

  int *nums = new int[N];

  srand(time(0));

  
  for (unsigned i = 0; i < N; i++) 
    nums[i] = rand() % RAND_RANGE_POPULATION;

  for (unsigned i = 0; i < N; i++) 
    cout << nums[i] << " ";
  cout << endl;

  quickSort(nums, 0, N -1);

  for (unsigned i = 0; i < N; i++)
    cout << nums[i] << " ";
  cout << endl;

  
  return 0;
}
