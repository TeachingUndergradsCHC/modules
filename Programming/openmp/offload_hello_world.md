# [Tutorial]: GPU Offloading with OpenMP: The Simplest Example   
[D2] Heterogeneous Programming with OpenMP  
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)


## Prerequisites 

This tutorial assumes we have a working environment for OpenMP-GCC target offloading. If you don't
have OpenMP target offloading set-up, the following tutorials may be helpful. 

  * [Setting up OpenMP-GCC target offloading on
  Ubuntu](https://github.com/apanqasem/tutorials/tree/main/openmp/gpu_offloading.md)
  * [Setting up OpenMP-GCC target offloading on Google
    Colab](https://colab.research.google.com/github/apanqasem/tutorials/tree/main/openmp/openmp_offload_colab.ipynb) 


#### Overview 

OpenMP allows code blocks in C/C++ and Fortran application to be offloaded to accelerators.  The
latest OpenMP supports offloading to AMD, NVIDIA and Intel GPUs. In this tutorial, we will focus on
NVIDIA GPUs only.   

 
#### The `target` Directive 

The `omp target` directive can be used to offload tasks to the GPU. The general format for the
`target` directive is shown below 

```C++

// ... code here executes on the CPU (host) 

// code in the succeeding block executes 
// on the GPU (device)
#pragma omp target      
{
   for (...)
     ...;
   ...
}

// ... code here executes on the CPU (host)
```

Using the template above we can execute any code block on the GPU. Consider the code below that scales the values in a floating-point array. 


```python
%%writefile gpu_hello_world.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float data[N];

  for (unsigned i = 0; i < N; i++) 
    data[i] = i;

  #pragma omp target
  for (unsigned i = 0; i < N; i++) 
    data[i] *= 3.14;
  
  cout << "Computation Done!" << endl; 
  
  // verify results
  for (unsigned i = 1; i < 2; i++) 
    cout << "data[1] = " << data[i] << endl;

  return 0;
}
```

    Overwriting gpu_hello_world.cpp


The `omp target` directive has been inserted around the `for` that performs the scaling. This will result in the `for` loop being executed on the GPU. The rest of the code will execute on the CPU. Let's try it out. 

To compile an OpenMP program with GPU offloading, we need to pass two additional flags: `-fno-stack-protector` `-foffload=nvptx-none`. See [Setting up NVIDIA GPU offloading for OpenMP-GCC on Ubuntu](https://github.com/TeachingUndergradsCHC/modules/blob/master/Algorithms/hybrid_algorithms/resources/gpu_offloading.md) for why that's necessary.


```python
!g++ -o gpu_hello_world gpu_hello_world.cpp -fno-stack-protector -foffload=nvptx-none -fopenmp
```

If you are doing this tutorial on your own machine and you get a compilation error, go through the set-up tutorials and make sure you have a CUDA-capable GPU that's being picked up by the NVIDIA driver and the device is "connected" to OpenMP. 

No extra steps are necessery to run an OpenMP application with offloading. So, we can run the application simply as follows 


```python
!./gpu_hello_world
```

    Computation Done!
    data[1] = 3.14


The code seems to be working. But do we know if the task was actually offloaded to the GPU? No! In certain cases, OpenMP may ignore the directive and just run the code on the host. To check that a GPU kernel is running we can profile the code with `nvprof`. This will tell us how much time is being spent on the GPU, if any. 


```python
!/usr/local/cuda/bin/nvprof ./gpu_hello_world
```

    ==24187== NVPROF is profiling process 24187, command: ./gpu_hello_world
    Computation Done!
    data[1] = 3.14
    ==24187== Profiling application: ./gpu_hello_world
    ==24187== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
     GPU activities:   99.88%  12.091ms         1  12.091ms  12.091ms  12.091ms  main$_omp_fn$0
                        0.07%  8.6720us         2  4.3360us     832ns  7.8400us  [CUDA memcpy HtoD]
                        0.05%  5.7280us         1  5.7280us  5.7280us  5.7280us  [CUDA memcpy DtoH]
          API calls:   72.92%  193.87ms         1  193.87ms  193.87ms  193.87ms  cuCtxCreate
                       17.93%  47.668ms         1  47.668ms  47.668ms  47.668ms  cuCtxDestroy
                        4.55%  12.098ms         1  12.098ms  12.098ms  12.098ms  cuCtxSynchronize
                        2.91%  7.7299ms         1  7.7299ms  7.7299ms  7.7299ms  cuModuleLoadData
                        0.95%  2.5219ms        20  126.10us  34.412us  1.1220ms  cuLinkAddData
                        0.24%  648.27us         1  648.27us  648.27us  648.27us  cuLaunchKernel
                        0.20%  524.89us         1  524.89us  524.89us  524.89us  cuLinkComplete
                        0.11%  298.19us         2  149.10us  147.82us  150.38us  cuMemAlloc
                        0.08%  208.32us         2  104.16us  103.93us  104.39us  cuMemFree
                        0.05%  125.47us        19  6.6030us     184ns  119.80us  cuDeviceGetAttribute
                        0.02%  48.638us         1  48.638us  48.638us  48.638us  cuLinkCreate
                        0.01%  38.050us         2  19.025us  12.509us  25.541us  cuMemcpyHtoD
                        0.01%  25.872us         1  25.872us  25.872us  25.872us  cuMemcpyDtoH
                        0.01%  24.104us         1  24.104us  24.104us  24.104us  cuDeviceGetName
                        0.01%  14.061us         2  7.0300us  2.2070us  11.854us  cuDeviceGetPCIBusId
                        0.00%  3.2090us         8     401ns     175ns     586ns  cuCtxGetDevice
                        0.00%  2.2340us         1  2.2340us  2.2340us  2.2340us  cuInit
                        0.00%  2.0950us         4     523ns     282ns     844ns  cuMemGetAddressRange
                        0.00%  1.9880us         4     497ns     179ns  1.1330us  cuDeviceGetCount
                        0.00%  1.7270us         1  1.7270us  1.7270us  1.7270us  cuLinkDestroy
                        0.00%  1.6750us         3     558ns     264ns     998ns  cuDeviceGet
                        0.00%  1.2010us         2     600ns     506ns     695ns  cuFuncGetAttribute
                        0.00%     847ns         1     847ns     847ns     847ns  cuModuleGetGlobal
                        0.00%     806ns         1     806ns     806ns     806ns  cuModuleGetFunction
                        0.00%     463ns         1     463ns     463ns     463ns  cuCtxGetCurrent
                        0.00%     196ns         1     196ns     196ns     196ns  cuDriverGetVersion


Indeed the `for` loop has been offloaded and run on the GPU for 12 milliseconds. This of course doesn't buy as any performance since we haven't actually parallelized the code and so we are not taking advantage of the GPU parallel resources. We can run the CPU-only version of the code for comparison.   


```python
%%writefile cpu_hello_world.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float data[N];

  for (unsigned i = 0; i < N; i++) 
    data[i] = i;

  for (unsigned i = 0; i < N; i++) 
    data[i] *= 3.14;
  
  cout << "Computation Done!" << endl; 
  
  // verify results
  for (unsigned i = 1; i < 2; i++) 
    cout << "data[1] = " << data[i] << endl;

  return 0;
}
```

    Writing cpu_hello_world.cpp



```python
!g++ -o cpu_hello_world cpu_hello_world.cpp -fopenmp
```


```python
!time ./cpu_hello_world
```

    Computation Done!
    data[1] = 3.14
    
    real	0m0.003s
    user	0m0.003s
    sys	0m0.000s


Not surprisingly, CPU is much faster. 

#### Offloading Parallel Code 

We almost never want to offload sequential tasks to the GPU. Any code that we want to offload to the GPU should be parallelized first. OpenMP makes this part easy as well. We can insert any `omp` pragma inside the offloaded region to paralellize the code (there are few exceptions, which we will discuss later). 

In our example, the `for` loop that scales the values in the `data` array can be parallelized with the `parallel for` pragma. 


```python
%%writefile gpu_hello_world_par.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float data[N];

  for (unsigned i = 0; i < N; i++) 
    data[i] = i;

  #pragma omp target
  #pragma omp parallel for 
  for (unsigned i = 0; i < N; i++) 
    data[i] *= 3.14;
  
  cout << "Computation Done!" << endl; 
  
  // verify results
  for (unsigned i = 1; i < 2; i++) 
    cout << "data[1] = " << data[i] << endl;

  return 0;
}
```

    Overwriting gpu_hello_world_par.cpp


In this version, the parallelized for loop is offloaded to the GPU. Let's if this makes a difference. Let us check the  


```python
!g++ -o gpu_hello_world_par gpu_hello_world_par.cpp -fno-stack-protector -foffload=nvptx-none -fopenmp
```


```python
!./gpu_hello_world_par
```

    Computation Done!
    data[1] = 3.14



```python
!/usr/local/cuda/bin/nvprof ./gpu_hello_world_par 2>&1 | grep main
```

     GPU activities:   99.15%  1.6688ms         1  1.6688ms  1.6688ms  1.6688ms  main$_omp_fn$0


The GPU kernel is now almost 12 times faster than before. This is version is also faster than the sequential CPU version. Let write a parallel version for the CPU (i.e., just take out the offload pragam).  


```python
%%writefile cpu_hello_world_par.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float data[N];

  for (unsigned i = 0; i < N; i++) 
    data[i] = i;

  #pragma omp parallel for 
  for (unsigned i = 0; i < N; i++) 
    data[i] *= 3.14;
  
  cout << "Computation Done!" << endl; 
  
  // verify results
  for (unsigned i = 1; i < 2; i++) 
    cout << "data[1] = " << data[i] << endl;

  return 0;
}
```

    Overwriting cpu_hello_world_par.cpp



```python
!g++ -o cpu_hello_world_par cpu_hello_world_par.cpp -fopenmp
```


```python
!time ./cpu_hello_world_par
```

    Computation Done!
    data[1] = 3.14
    
    real	0m0.008s
    user	0m0.052s
    sys	0m0.000s


On the CPU, the parallel version is actually slower than sequential. _Can we explain this behavior?_

We will notice that in this example, each thread is doing very little work (just one multiplication and store). On the CPU, which consists of complex powerful processing cores, this type of fine-grain parallelism often does not yield good results. The GPU on the other hand consists of _many_ simple cores and can execute this type of parallel code more efficiently. 

#### Summary

We have learned how to offload parallel tasks to GPU using the `target` directive. This example is very simple, however. In this example, we let OpenMP make all the decisions about data mappin and thread creation and scheduling. To get better performance out of GPUs we will want control over these. We will look at the various clauses associated with the `target` that provides mechansims for explicit data mapping and creation of teams of threads. 
