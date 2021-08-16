/*
 * Example image processing program to remove red color using SYCL
 * and translated from the CUDA program
 *
 * References:
 * https://www.codeproject.com/Articles/5284847/5-Minutes-to-Your-First-oneAPI-App-on-DevCloud
 * 
 * The file stb_image_write.h can be obtained from
 * https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
 * The file stb_image.h can be obtained from
 * https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
 *
 * compile with:
 *    syclcc -o noRed noRed.cpp
 * run with:
 *    ./noRed
 *
 */

#include <CL/sycl.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main (int argc, char *argv[]){
    const char* inFile = "640x426.bmp";     //file names for input and output files
    const char* outFile = "out.bmp";

    int width;                              //image size
    int height;
    int channels;
    uint8_t* inImage = stbi_load(inFile, &width, &height, &channels, 3);
    int image_size = width * height * channels;    //size of image in byes; 3 is # channels

    // start device queue
    cl::sycl::default_selector deviceSelector;
    cl::sycl::queue queue(deviceSelector);

    // Create device memory
    cl::sycl::buffer<uint8_t, 1> d_input(inImage , image_size);
    // Allocate shared memory between device and host
    uint8_t* output_data = reinterpret_cast<uint8_t*>
    (cl::sycl::malloc_shared(image_size, queue));
    queue.submit([output_data, &d_input, width, height](
    cl::sycl::handler& cgh) {
      auto input_data = d_input.get_access
      <cl::sycl::access::mode::read>(cgh);
      cgh.parallel_for(cl::sycl::range<1>(width * height),
        [input_data, output_data](cl::sycl::id<1> idx){
	int offset = 3*idx[0];
	// Red channel
	output_data[offset] = 0;
	// Green channel
	output_data[offset+1] = input_data[offset+1];
	// Blue channel
	output_data[offset+2] = input_data[offset+2];
	});
      });
    queue.wait();
    stbi_write_bmp(outFile, width, height, 3, output_data); //write output image to file
    stbi_image_free(inImage);               //free memory
    cl::sycl::free(output_data, queue);

    return 0;
}
