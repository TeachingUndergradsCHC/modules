/*
 * Sample program to illustrate threads and blocks in CUDA.
 *
 * compile with:
 *    nvcc -o hello hello.cu
 * run with:
 *    ./hello
 */
 
#include <stdio.h>
 
__global__ void hello() {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello from thread %d (%d of block %d)\n", id, threadIdx.x, blockIdx.x);
}

int main() {
  hello<<<3,4>>>();  //launch 3 blocks of 4 threads each

  cudaDeviceSynchronize();  //make sure kernel completes
}
