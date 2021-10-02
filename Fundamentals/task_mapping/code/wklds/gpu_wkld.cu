/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


/* 
 * Code to simulate a GPU workload for lab assignment in [A2] Task Mapping on Soft Heterogeneous Systems. 
 * Workload consists of a the Black-Scholes kernel taken from NVIDIA SDK 10.1
 * 
 * Computation is done on the GPU when the user selects a core attached to a GPU; otherwise the code is run on 
 * the CPU. GPU version of the code is expected to run faster.  
 *
 * @author: Apan Qasem <apan@txstate.edu>
 * @date: 04/02/20 
 * 
 * @update: 03/12/21
 */
#include<cstdio>
#include<sys/time.h>  


///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
static double CND(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(
    float &callResult,
    float &putResult,
    float Sf, //Stock price
    float Xf, //Option strike
    float Tf, //Option years
    float Rf, //Riskless rate
    float Vf  //Volatility rate
)
{
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    callResult   = (float)(S * CNDD1 - X * expRT * CNDD2);
    putResult    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    for (int opt = 0; opt < optN; opt++)
        BlackScholesBodyCPU(
            h_CallResult[opt],
            h_PutResult[opt],
            h_StockPrice[opt],
            h_OptionStrike[opt],
            h_OptionYears[opt],
            Riskfree,
            Volatility
        );
}


// extern "C" void BlackScholesCPU(
//     float *h_CallResult,
//     float *h_PutResult,
//     float *h_StockPrice,
//     float *h_OptionStrike,
//     float *h_OptionYears,
//     float Riskfree,
//     float Volatility,
//     int optN
// );

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = sqrtf(T);
    d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;

    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    //for (int opt = tid; opt < optN; opt += THREAD_N)
    if (opt < optN)
        BlackScholesBodyGPU(
            d_CallResult[opt],
            d_PutResult[opt],
            d_StockPrice[opt],
            d_OptionStrike[opt],
            d_OptionYears[opt],
            Riskfree,
            Volatility
        );
}


////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int  NUM_ITERATIONS = 512;


const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  if (argc < 3) {
    fprintf(stderr, "usage: ./blackscholes options GPU\n");
    exit(0);
  }
  unsigned options = atoi(argv[1]);
  int options_size = options * sizeof(float);
  unsigned gpu = atoi(argv[2]);
  
  float
    *h_CallResultCPU,
    *h_PutResultCPU,
    *h_CallResultGPU,
    *h_PutResultGPU,
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    float
    *d_CallResult,
    *d_PutResult,
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    double delta, ref, sum_delta, sum_ref, max_delta, L1norm;

    int i;


    // CPU memory allocation
    h_CallResultCPU = (float *)malloc(options_size);
    h_PutResultCPU  = (float *)malloc(options_size);
    h_CallResultGPU = (float *)malloc(options_size);
    h_PutResultGPU  = (float *)malloc(options_size);
    h_StockPrice    = (float *)malloc(options_size);
    h_OptionStrike  = (float *)malloc(options_size);
    h_OptionYears   = (float *)malloc(options_size);

    // GPU memory allocation
    cudaMalloc((void **)&d_CallResult,   options_size);
    cudaMalloc((void **)&d_PutResult,    options_size);
    cudaMalloc((void **)&d_StockPrice,   options_size);
    cudaMalloc((void **)&d_OptionStrike, options_size);
    cudaMalloc((void **)&d_OptionYears,  options_size);

    srand(5347);

    // Generate options set
    for (i = 0; i < options; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    // Copy options data to GPU memory for further processing
    cudaMemcpy(d_StockPrice,  h_StockPrice,   options_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  options_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   options_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    timeval starttime, endtime;
    double runtime; 
    if (gpu) {
      gettimeofday(&starttime, NULL);
      
      for (i = 0; i < NUM_ITERATIONS; i++) {
	BlackScholesGPU<<<DIV_UP(options, 128), 128>>>(
						       d_CallResult,
						       d_PutResult,
						       d_StockPrice,
						       d_OptionStrike,
						       d_OptionYears,
						       RISKFREE,
						       VOLATILITY,
						       options
						       );
      }
      
      
      cudaDeviceSynchronize();
      gettimeofday(&endtime, NULL);
    }
    else {
      gettimeofday(&starttime, NULL);
      //Calculate options values on CPU
      BlackScholesCPU(
		      h_CallResultCPU,
		      h_PutResultCPU,
		      h_StockPrice,
		      h_OptionStrike,
		      h_OptionYears,
		      RISKFREE,
		      VOLATILITY,
		      options
		      );
      
      gettimeofday(&endtime, NULL);
    }
    // Read back GPU results to compare them to CPU results
    cudaMemcpy(h_CallResultGPU, d_CallResult, options_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  options_size, cudaMemcpyDeviceToHost);


    runtime = endtime.tv_sec + endtime.tv_usec / 1000000.0 - (starttime.tv_sec + starttime.tv_usec / 1000000.0); 
    fprintf(stdout, "\033[1;32m[wk3] compute time = %.3f s\n\033[0m", runtime);
#ifdef VERIFY
    printf("%3.5f,%3.5f\n", h_CallResultGPU[2047],h_PutResultGPU[3145]);  
#endif
    
    // validation not use; code is running on either GPU or CPU 
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;

    for (i = 0; i < options; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;

    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);

    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);


    if (L1norm > 1e-6)
    {
      exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}
