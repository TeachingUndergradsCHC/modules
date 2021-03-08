## Setting up NVIDIA GPU offloading for OpenMP-GCC on Ubuntu

### Description 

This tutorial gives a step-by-step instruction for setting up the environment for target offloading
on to NVIDIA GPUs. The instructions are specifically for Ubuntu systems. 

Target offloading has been natively supported on GCC since version 7. 
GCC supports various types of offloading including offloading to Intel MIC and AMD GCN. This
tutorial focuses on NVIDIA GPUs only
The target construct was introduced in OpenMP 4.0 with significant improvements in OpenMP 4.5

### Outline 

  * [System Requirements](#sys)
  * [GCC](#gcc)
  * [OpenMP](#openmp)
  * [CUDA](#cuda)
  * [Potential Pitfalls](#pitfalls)


### <a name="sys"></a>System Requirements

 * Ubuntu 18.04 LTS. Ubuntu 16.04 and Ubuntu-based distributions like Kubuntu, Linux Mint and
   Elementary OS should work. But not tested. 
 * NVIDIA GPU with CUDA support: Kepler, Maxwell, Pascal, Volta 
 * CPU: Intel, AMD, POWER
 * sudo access is preferred, local installations will get messy. 
 


### <a name="gcc"></a>GCC

Ubuntu 18.04 is distributed with GCC 7. Although offloading is supported it is better to upgrade to
a newer version. I recommend upgrading to GCC 8 (see pitfall \# 2). 

```bash
sudo apt install gcc-8 g++-8 gfortran-8
```

GCC distribution with Ubuntu doesn't automatically support task offloading. For this we need to
install another package. 

```bash
sudo apt install gcc-8-offload-nvptx
```

The upgrade does not need to clobber the default GCC or older versions of GCC. The
`update-alternatives` utility can maintain multiple distributions of GCC (and other packages)
simultaneously and cab be used to select one of the alternatives at any time. 

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++
/usr/bin/g++-8
```

### <a name="openmp"></a>OpenMP

Although there is support for target offload in OpenMP 4.0. The support is minimal. We will need
at least 4.5. GCC 8 comes with OpenMP 4.5. The program below will verify the version of OpenMP
integrated with GCC. 

```C++
#include <unordered_map>
#include <cstdio>
#include <omp.h>

int main(int argc, char *argv[]) {
  std::unordered_map<unsigned,std::string> map{
    {200505,"2.5"},{200805,"3.0"},{201107,"3.1"},{201307,"4.0"},{201511,"4.5"}};
  printf("We have OpenMP %s.\n", map.at(_OPENMP).c_str());
  return 0;
}
```

We can then build and run the code to get the version number. 

```bash
(ada)% g++ -o omp_version omp_version.cpp -fopenmp
(ada)% ./omp_version 
We have OpenMP 4.5.
```

Getting the latest OpenMP version is best because that provides all the features. However, that
installation is problematic. 

### <a name="cuda"></a>CUDA

Target offloading will work with CUDA driver/runtime version 7.0 or above. But it's best to upgrade
to the latest or close to the latest CUDA driver. To upgrade follow these instructions. 

### <a name="test"></a>Testing

We can use the following code to test if offloading is working. 


```bash
g++ -o omp_offload_test0 -fopenmp -O3 omp_offload_test0.cpp  -fno-stack-protector -foffload=nvptx-none
nvprof ./omp_offload_test0
```


### <a name="pitfalls"></a>Potential Pitfalls
   
Many resources out there that show how to set up gcc for task offloading. Most of these are based on
building GCC and associated tools from sources. There is no need for this. 

nvcc can't compile with GNU libraries greater > 8. 

#### The stack-protector problem 

```bash
ptxas /tmp/ccq6t6e2.o, line 189; error   : Illegal operand type to instruction 'ld'
ptxas /tmp/ccq6t6e2.o, line 246; error   : Illegal operand type to instruction 'ld'
ptxas /tmp/ccq6t6e2.o, line 189; error   : Unknown symbol '__stack_chk_guard'
ptxas /tmp/ccq6t6e2.o, line 246; error   : Unknown symbol '__stack_chk_guard'
ptxas fatal   : Ptx assembly aborted due to errors
nvptx-as: ptxas returned 255 exit status
mkoffload: fatal error: x86_64-linux-gnu-accel-nvptx-none-gcc-9 returned 1 exit status
compilation terminated.
lto-wrapper: fatal error: /usr/lib/gcc/x86_64-linux-gnu/9//accel/nvptx-none/mkoffload returned 1 exit status
compilation terminated.
/usr/bin/ld: error: lto-wrapper failed
collect2: error: ld returned 1 exit status
```


#### offloading Support not installed 


```bash 
lto-wrapper: fatal error: could not find accel/nvptx-none/mkoffload in
/usr/lib/gcc/x86_64-linux-gnu/9/:/usr/lib/gcc/x86_64-linux-gnu/9/:/usr/lib/gcc/x86_64-linux-gnu/:/usr/lib/gcc/x86_64-linux-gnu/9/:/usr/lib/gcc/x86_64-linux-gnu/
(consider using ‘-B’)
```

### References 
 * [How to Install GCC Compiler on Ubuntu 18.04](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/)
 * [Stackoverflow:OpenMP offloading with GCC fails](https://stackoverflow.com/questions/62855136/openmp-offloading-with-gcc-fails-with-ptx-assembly-aborted-due-to-errors) 
 * [Stackoverflow: OpenMP offloading says 'fatal error: could not find
   accel/nvptx-none/mkoffload'](https://stackoverflow.com/questions/62855087/openmp-offloading-says-fatal-error-could-not-find-accel-nvptx-none-mkoffload) 
   






