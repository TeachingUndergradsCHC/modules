#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "/content/ppmFile.c"

__global__ void kernel(int width, int height, unsigned char *d_input, unsigned char* d_output){

    //coordinates of pixel for which this call is responsible
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int offset;  //index in array corresponding to a pixel

    if(i >=0 && i < width && j >=0 && j < height) {

          offset = (j * width + i) * 3 + 0;  //0 is red channel
          d_output[offset] = 0;

          offset = (j * width + i) * 3 + 1;  //1 is green channel
          d_output[offset] = d_input[offset];

          offset = (j * width + i) * 3 + 2;  //2 is blue channel
          d_output[offset] = d_input[offset];
    }
}

#include "/content/main.c"
