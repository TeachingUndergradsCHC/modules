### Moving Data Between Host and Device

#### Objective 

This tutorial shows how we can use the `map` clause with the `target data` directive to copy data to and from the GPU. 

#### Overview 
The CPU and GPU have separate memory spaces. When we want the GPU to access data allocated on the CPU (or _vice versa_), we need to copy the data from one memory to the other. By default OpenMP will copy all variables within lexical scope to and from the device. Notwithstanding, in certain cases we need to tell OpenMP which data we want copied. 

The code below performs a vector addition. The `target` directive has been used to offload the vector addition task to the GPU. 


```python
%%writefile vec_add.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float A[N];
  float B[N];
  float C[N];

  for (unsigned i = 0; i < N; i++) {
    A[i] = i * 2.17;
    B[i] = i * 3.14;
    C[i] = 0;
  }

  #pragma omp target
  {  
    #pragma omp parallel for 
    for (unsigned i = 0; i < N; i++)
      C[i] = A[i] + B[i];
  }

  cout << "Computation Done!" << endl;

  // verify results                                                                                           
  for (unsigned i = 1; i < 2; i++)
    cout << "C[1] = " << C[i] << endl;

  return 0;
}
```

    Overwriting vec_add.cpp


Let's compile and run this code. 


```python
!g++ -o vec_add -fopenmp vec_add.cpp  -fno-stack-protector -foffload=nvptx-none -fopenmp
```


```python
!perf stat ./vec_add
```

    Computation Done!
    C[1] = 5.31
    
     Performance counter stats for './vec_add':
    
                224.66 msec task-clock                #    0.850 CPUs utilized          
                    58      context-switches          #    0.258 K/sec                  
                     0      cpu-migrations            #    0.000 K/sec                  
                 3,472      page-faults               #    0.015 M/sec                  
           629,850,790      cycles                    #    2.804 GHz                    
           606,730,402      instructions              #    0.96  insn per cycle         
           128,837,039      branches                  #  573.471 M/sec                  
             2,773,679      branch-misses             #    2.15% of all branches        
    
           0.264281673 seconds time elapsed
    
           0.068586000 seconds user
           0.157345000 seconds sys
    
    


#### The `target data` construct 
The example code uses static allocation for the array A, B, and C. This is very limiting and will rarely appear in practice. Let's modify the code to do dynamic allocation of A. We will save this version of the code as `vec_add_dynamic.cpp`  


```python
%%writefile vec_add_dynamic.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float *A = (float *) malloc(sizeof(float) * N);
  float B[N];
  float C[N];

  for (unsigned i = 0; i < N; i++) {
    A[i] = i * 2.17;
    B[i] = i * 3.14;
    C[i] = 0;
  }

  #pragma omp target 
  {
  
    #pragma omp parallel for 
    for (unsigned i = 0; i < N; i++)
      C[i] = A[i] + B[i];
  }

  cout << "Computation Done!" << endl;

  // verify results                                                                                           
  for (unsigned i = 1; i < 2; i++)
    cout << "C[1] = " << C[i] << endl;

  return 0;
}
```

    Overwriting vec_add_dynamic.cpp


Let's compile and run this version.


```python
!g++ -o vec_add -fopenmp vec_add_dynamic.cpp  -fno-stack-protector -foffload=nvptx-none -fopenmp
```


```python
!./vec_add
```

    
    libgomp: cuCtxSynchronize error: an illegal memory access was encountered
    
    libgomp: cuMemFree_v2 error: an illegal memory access was encountered
    
    libgomp: device finalization failed


**_What happened?_**  The error message indicates that the GPU kernel is trying to access data that has not been allocated to GPU memory. Why did OpenMP not copy the `A` array? By default, OpenMP will copy, both to and from the device, all scalar variables and static arrays in scope. However, it will not copy dynamically allocated data. (The [OpenMP 4.5 specs](https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf) is a little unclear about this). 

So, we need to tell OpenMP to create copy 'A' to device. 
Essentially, this is telling OpenMP that we are doing dynamic memory allocation, so make sure there is enough space. And this is why we are getting the error. 


#### The `map` clause 
The `map` clause is used to explicitly map data to device memory. `map` takes a list of variables as its arugment and maps them to device memory. An optional qualifier can be specified to control _how_ data is mapped. More on this latrer. For this example, we want to map A to device memory. When mapping dyanamically allocated data, the number of elements that need to be mapped must also be specified. Bad things will happen otheriwse.


```python
%%writefile vec_add_dynamic.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float *A = (float *) malloc(sizeof(float) * N);
  float B[N];
  float C[N];

  for (unsigned i = 0; i < N; i++) {
    A[i] = i * 2.17;
    B[i] = i * 3.14;
  }

  #pragma omp target map(A[0:N])
  {
    #pragma omp parallel for 
    for (unsigned i = 0; i < N; i++)
      C[i] = A[i] + B[i];
  }

  cout << "Computation Done!" << endl;

  // verify results                                                                                           
  for (unsigned i = 1; i < 2; i++)
    cout << "C[1] = " << C[i] << endl;

  return 0;
}
```

    Overwriting vec_add_dynamic.cpp


Let's try out the corrected version. 


```python
!g++ -o vec_add -fopenmp vec_add_dynamic.cpp  -fno-stack-protector -foffload=nvptx-none -fopenmp
```


```python
! /usr/local/cuda/bin/nvprof ./vec_add 2>&1 | grep main
```

     GPU activities:   97.64%  1.6623ms         1  1.6623ms  1.6623ms  1.6623ms  main$_omp_fn$0


Now, let's allocate _B_ and _C_ in dynamic memory and add the appropiate `map` clauses. Note, we are still relying on OpenMP to implicitly map _N_


```python
%%writefile vec_add_dynamic.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float *A = (float *) malloc(sizeof(float) * N);
  float *B = (float *) malloc(sizeof(float) * N);
  float *C = (float *) malloc(sizeof(float) * N);

  for (unsigned i = 0; i < N; i++) {
    A[i] = i * 2.17;
    B[i] = i * 3.14;
  }

  #pragma omp target map(A[0:N],B[0:N],C[0:N])
  {
    #pragma omp parallel for 
    for (unsigned i = 0; i < N; i++)
      C[i] = A[i] + B[i];
  }

  cout << "Computation Done!" << endl;

  // verify results                                                                                           
  for (unsigned i = 1; i < 2; i++)
    cout << "C[1] = " << C[i] << endl;
  
  return 0;
}
```

    Overwriting vec_add_dynamic.cpp



```python
!g++ -o vec_add -fopenmp vec_add_dynamic.cpp  -fno-stack-protector -foffload=nvptx-none -fopenmp
```


```python
!/usr/local/cuda/bin/nvprof ./vec_add 2>&1 | grep "main\|HtoD\|DtoH"
```

     GPU activities:   98.70%  1.6634ms         1  1.6634ms  1.6634ms  1.6634ms  main$_omp_fn$0
                        0.96%  16.097us         3  5.3650us     832ns  7.6800us  [CUDA memcpy HtoD]
                        0.34%  5.7930us         1  5.7930us  5.7930us  5.7930us  [CUDA memcpy DtoH]
                        0.02%  60.233us         3  20.077us  12.680us  25.180us  cuMemcpyHtoD
                        0.02%  51.392us         1  51.392us  51.392us  51.392us  cuMemcpyDtoH


#### Controlling `map` behavior 

Be default OpenMP will copy all mapped data from the CPU to the GPU at the beginning of the offloaded and then copy everything back at the end of the taks. This may lead to many unnecessary copies. We can optimize this behavior with modifiers in the `map` clause. 

The `map` clause accepts a modifier that allows us to specify the direction of data movement. In the code below, the says that the _A_ and _B_ arrays should be copied _to_ device memory when executing the offloaded task while the _C_ array should be copied _from_ device memory to host memory. For the vector add computation, this makes sense. The initialized values in _A_ and _B_ are copied to the GPU. We do not need to copy them back since the GPU doesn't modify these arrays. On the other hand, we do not need to copy _C_ to GPU but we did not to copy it back to the CPU to access the values updated by the GPU. 

If there is data that is both read from and written to by the GPU, we can just use `tofrom` modifier. 


```python
%%writefile vec_add_dynamic.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  float *A = (float *) malloc(sizeof(float) * N);
  float *B = (float *) malloc(sizeof(float) * N);
  float *C = (float *) malloc(sizeof(float) * N);

  for (unsigned i = 0; i < N; i++) {
    A[i] = i * 2.17;
    B[i] = i * 3.14;
  }

  #pragma omp target map(to:A[0:N],B[0:N]) map(from:C[0:N])
  {
     #pragma omp parallel for 
    for (unsigned i = 0; i < N; i++)
      C[i] = A[i] + B[i];
  }

  cout << "Computation Done!" << endl;

  // verify results                                                                                           
  for (unsigned i = 1; i < 2; i++)
    cout << "C[1] = " << C[i] << endl;
  
  return 0;
}
```

    Overwriting vec_add_dynamic.cpp



```python
!g++ -o vec_add -fopenmp vec_add_dynamic.cpp  -fno-stack-protector -foffload=nvptx-none -fopenmp
```


```python
!/usr/local/cuda/bin/nvprof ./vec_add 2>&1 | grep "main"
```

     GPU activities:   98.70%  1.6626ms         1  1.6626ms  1.6626ms  1.6626ms  main$_omp_fn$0


##### Summary 

In this tutorial we saw how we can use the `map` clause to copy data to and from the GPU device when executing an offloaded task. The `map` clause is necessary whenever we are accessing dynamically allocated data. For static data structures and scalar variables, OpenMP will do the mapping implicitly. 


```python

```
