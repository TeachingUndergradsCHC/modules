/*
 * Sample program that uses SYCL to perform element-wise add of two
 * vectors.  Each element is the responsibility of a separate thread.
 * 
 *  Based on the example CUDA program and on
 *  https://github.com/illuhad/hipSYCL/blob/develop/doc/examples.md
 *
 * compile with:
 *    syclcc -o addVectors addVectors.cpp
 * run with:
 *    ./addVectors
 */

#include <iostream>
#include <CL/sycl.hpp>
#include <chrono>

using data_type = int;

//problem size (vector length):
#define N 10

std::vector<data_type> kernel(cl::sycl::queue& q,
	       const std::vector<data_type>& a,
               const std::vector<data_type>& b) {
  //function that runs on GPU/CPU to do the addition

	std::vector<data_type> c(a.size());
	cl::sycl::range<1> work_items{a.size()};
	{
		// copy data to compute device
		cl::sycl::buffer<data_type> buff_a(a.data(), a.size());
		cl::sycl::buffer<data_type> buff_b(b.data(), b.size());
		cl::sycl::buffer<data_type> buff_c(c.data(), c.size());
                // sets c[i] = a[i] + b[i]
		// each thread is responsible for one value of i
		q.submit([&](cl::sycl::handler &cgh){
			auto access_a = buff_a.get_access<cl::sycl::access::mode::read>(cgh);
			auto access_b = buff_b.get_access<cl::sycl::access::mode::read>(cgh);
			auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);
			cgh.parallel_for<class vector_add>(work_items,
					[=] (cl::sycl::id<1> tid) {
					access_c[tid] = access_a[tid] + access_b[tid];
					});
		});
	}
	return c;
}

int main() {
  std::vector<data_type> a(N);       //input arrays
  std::vector<data_type> b(N);
  std::vector<data_type> res(N);     //output array

  //setup command queue
  cl::sycl::queue q;

  //set up contents of a and b
  for(int i=0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }

  //start timer
  auto start = std::chrono::high_resolution_clock::now();

  //call the kernel
  res = kernel(q, a, b);

  //stop timer and print time
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli>  diff = stop - start;
  std::cout << "time: " << diff.count() << " ms" << std::endl;

  //verify results
  for(int i=0; i < N; i++)
    std::cout << res[i] << std::endl;
}
