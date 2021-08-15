/*
 * Sample program to illustrate global index and local groups in SYCL.
 * In comparison to CUDA which uses blocks and threads within a block
 * so that the total work items are given by number of blocks times
 * number of threads, in SYCL the total work items and number of
 * blocks of work are given and from this the work per block can be
 * calculated.
 *
 * This example is modified from the CUDA version using the SYCL 
 * documentation
 * https://sycl.readthedocs.io/en/latest/iface/stream.html
 * and the example codes at
 * https://www.codingame.com/playgrounds/48226/introduction-to-sycl/debugging-sycl-applications-2
 * https://developer.codeplay.com/products/computecpp/ce/guides/sycl-for-cuda-developers/examples
 * https://github.com/illuhad/hipSYCL/blob/develop/doc/scoped-parallelism.md
 * https://stackoverflow.com/questions/58437021/using-barriers-in-sycl
 * https://github.com/illuhad/hipSYCL/blob/develop/tests/sycl/group_functions/group_functions.hpp
 *
 * compile with:
 *     syclcc -o hello hello.cpp
 * run with:
 *    ./hello
 */
 
#include <CL/sycl.hpp>

void hello(cl::sycl::queue& q,
	   const size_t& a, 
	   const size_t& b) {
	   q.submit([&](cl::sycl::handler& cgh) {
	     auto out = cl::sycl::stream(1024, 128, cgh);
	     cgh.parallel_for<class hello>(
	       cl::sycl::nd_range<1>{a, b},
               [=] (cl::sycl::nd_item<1> item) {	  
	       auto group = item.get_group();
	       auto global_id = item.get_global_linear_id();
	       auto local_id = item.get_local_linear_id();
	       // setup sycl stream class to print local index in work group and 
	       // global index in work group
	       out << "Hello from " << group << " with global id " << global_id
	           << " and local id " << local_id << cl::sycl::endl;
	     });
	   q.wait_and_throw();
	   });
}

int main() {
	cl::sycl::queue q;
	size_t a = 12;
	size_t b = 4;
      	hello(q,a,b);  //launch a global size of 12 with a local size of 4
	return 0;
}
