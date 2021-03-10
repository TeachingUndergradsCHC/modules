/*
 * Sample program that uses CUDA to perform element-wise add of two
 * vectors.  Each element is the responsibility of a separate thread.
 *
 * compile with:
 *    nvcc -o addVectors addVectors.cu
 * run with:
 *    ./addVectors
 */

#include <stdio.h>

//problem size (vector length):
#define N 10

__global__ void kernel(int* res, int* a, int* b) {
  //function that runs on GPU to do the addition
  //sets res[i] = a[i] + b[i]; each thread is responsible for one value of i

  int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
  if(thread_id < N) {
    res[thread_id] = a[thread_id] + b[thread_id];
  }
}

void check(cudaError_t retVal) {
  //takes return value of a CUDA function and checks if it was an error
  if(retVal != cudaSuccess) {
    fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(retVal));
    exit(1);
  }
}

int main() {
  int* a;       //input arrays (on host)
  int* b;
  int* res;     //output array (on host)

  int* a_dev;   //input arrays (on GPU)
  int* b_dev;
  int* res_dev; //output array (on GPU) 

  //allocate memory
  a = (int*) malloc(N*sizeof(int));
  b = (int*) malloc(N*sizeof(int));
  res = (int*) malloc(N*sizeof(int));
  check(cudaMalloc((void**) &a_dev, N*sizeof(int)));
  check(cudaMalloc((void**) &b_dev, N*sizeof(int)));
  check(cudaMalloc((void**) &res_dev, N*sizeof(int)));

  //set up contents of a and b
  for(int i=0; i < N; i++)
    a[i] = b[i] = i;

  //allocate timers
  cudaEvent_t start;
  check(cudaEventCreate(&start));
  cudaEvent_t stop;
  check(cudaEventCreate(&stop));

  //start timer
  check(cudaEventRecord(start,0));

  //transfer a and b to the GPU
  check(cudaMemcpy(a_dev, a, N*sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMemcpy(b_dev, b, N*sizeof(int), cudaMemcpyHostToDevice));

  //call the kernel
  int threads = 512;                   //# threads per block
  int blocks = (N+threads-1)/threads;  //# blocks (N/threads rounded up)
  kernel<<<blocks,threads>>>(res_dev, a_dev, b_dev);

  //transfer res to the host
  check(cudaMemcpy(res, res_dev, N*sizeof(int), cudaMemcpyDeviceToHost));

  //stop timer and print time
  check(cudaEventRecord(stop,0));
  check(cudaEventSynchronize(stop));
  float diff;
  check(cudaEventElapsedTime(&diff, start, stop));
  printf("time: %f ms\n", diff);

  //deallocate timers
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));

  //verify results
  for(int i=0; i < N; i++)
    printf("%d ", res[i]);
  printf("\n");

  //free the memory
  free(a);
  free(b);
  free(res);
  check(cudaFree(a_dev));
  check(cudaFree(b_dev));
  check(cudaFree(res_dev));
}
