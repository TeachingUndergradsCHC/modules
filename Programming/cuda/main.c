int main (int argc, char *argv[]){
    const char* inFile = "640x426.ppm";     //file names for input and output files
    const char* outFile = "out.ppm";

    int width;                              //image size
    int height;
    Image *inImage, *outImage;              //image structs (defined in ppmFile.h)
    unsigned char *data;                    //input image data

    //Device variables:
    unsigned char *d_input;                 //input image data
    unsigned char *d_output;                //output image data

    inImage = ImageRead(inFile);            //get input image and its attributes  
    width = inImage->width;
    height = inImage->height;
    data = inImage->data;
    int image_size = width * height * 3;    //size of image in byes; 3 is # channels

    //allocate memory for GPU
    cudaMalloc((void**)&d_input, sizeof(unsigned char*) * image_size);
    cudaMalloc((void**)&d_output, sizeof(unsigned char*) * image_size);

    //copy values to GPU
    cudaMemcpy(d_input, data, image_size, cudaMemcpyHostToDevice);

    //call kernel using block size 32x32
    dim3 blockD(32,32);
    dim3 gridD((width + blockD.x - 1)/blockD.x, (height + blockD.y - 1)/blockD.y);
    kernel<<<gridD, blockD>>>(width, height, d_input,d_output);
    
    //create and clear image variable for use as the result
    outImage = ImageCreate(width,height);
    ImageClear(outImage,255,255,255);
    
    cudaDeviceSynchronize();

    //copy output image from gpu
    cudaMemcpy(outImage->data, d_output, image_size, cudaMemcpyDeviceToHost);

    ImageWrite(outImage, outFile);        //write output image to file

    free(inImage->data);                  //free memory
    free(outImage->data);

    return 0;
}
