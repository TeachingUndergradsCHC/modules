### Setting up NVIDIA GPU offloading for OpenMP-GCC on Ubuntu
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)

### Description 

This tutorial gives step-by-step instructions for setting up the environment to do target offloading
on to NVIDIA GPUs with OpenMP and GCC. The instructions are for Ubuntu.

The target construct, which allows structured code blocks to be offloaded to accelerators, was first
introduced in OpenMP 4.0 with significant improvements in OpenMP 4.5 and OpenMP 5.0. GCC has supported
OpenMP's target offload feature since version 7 (7.2.0, I think). GCC supports offloading
including to Intel MIC, and NVIDIA and AMD GCN. This tutorial focuses on offloading to NVIDIA GPUs
only. 

### Outline 

  * [Pre-requisites](#sys)
  * [GCC](#gcc)
  * [OpenMP](#openmp)
  * [CUDA](#cuda)
  * [Potential Pitfalls](#pitfalls)


### <a name="sys"></a>Pre-requisites

 * **Ubuntu 18.04 LTS:** Ubuntu 16.04 and Ubuntu-based distributions like Kubuntu, Linux Mint and
   Elementary OS should work. But not tested. 
 * **NVIDIA GPU with CUDA support:** Tested with Kepler, Maxwell, Pascal, Volta. Older or
   newer systems may not work. 
 * **CPU:** Tested on Intel Core and Xeon, AMD Threadripper and POWER8
 * **sudo access:** sudo access is preferred; local installation is possible but messy. 
 
### <a name="gcc"></a>GCC

#### Upgrading GCC
Ubuntu 18.04 is distributed with GCC 7. Although offloading is supported in GCC 7,
it is better to upgrade to a newer version (but not the latest, see pitfall \# 2). I recommend
upgrading to GCC 8.  

To install GCC 8 with `apt`. 

```bash
sudo apt install gcc-8 g++-8 gfortran-8
```

#### Enabling offloading

Ubuntu GCC distributions are not configured to automatically support target offloading. To enable
offloading for GCC 8, we need to install the appropriate `offload` package. 

```bash
sudo apt install gcc-8-offload-nvptx
```

#### Maintaining multiple versions of GCC

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
g++ -o omp_version omp_version.cpp -fopenmp
./omp_version 
We have OpenMP 4.5.
```

Getting the latest OpenMP version is best because that provides all the features. However, that
installation is problematic. 

### <a name="cuda"></a>CUDA

Target offloading will work with CUDA driver/runtime version 7.0 or above. But it's best to upgrade
to the latest CUDA driver. To upgrade to the latest CUDA and CUDA drivers type the following. 

```bash
sudo apt-get install cuda
sudo apt-get install cuda-drivers
```

### <a name="test"></a>Testing

We can use the following simple code to test if offloading is working. 

```C++ 
#include <iostream>

using namespace std;

int main(){
  const int N = 1000;
  int a[N];

  for(unsigned i = 0; i < N; i++)
    a[i] = 1;

  #pragma omp target 
  for(unsigned i = 0;i < N; i++)
    a[i] *= 3 * i + 1;

  for(unsigned i = 0; i < 1; i++)
    cout << "Result a[0] = " << a[i] << endl;
  return 0;
}
```

To build the above program with offloading 

```bash
g++ -o omp_offload_test0 -fopenmp -O3 omp_offload_test0.cpp  -fno-stack-protector -foffload=nvptx-none
```

To run the code and verify that work is being offloaded 

```
nvprof ./omp_offload_test0
```




### <a name="pitfalls"></a>Potential Pitfalls
   
#### 1. Building GCC from source

There are many tutorials and resources out there for setting up GCC for task offloading. Most of
these require building GCC and associated GNU tools from source, which can get hairy. There is no
need to build from source as long as you have a relatively recent Linux distribution. 

#### 2. The stack-protector problem 

On some systems you _must_ pass the `-fno-stack-protector` when compiling code for
offloading. Otherwise you will get a build error like the following. 
```bash
g++ -o omp_offload_test0 -fopenmp -O3 omp_offload_test0.cpp -foffload=nvptx-none
```

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

It is fine, AFAIK, to pass the `-fno-stack-protector` even if the system does not require it. So to
be safe, you can always include `-fno-stack-protector`

#### 3. Offloading support not installed 

If you have upgraded GCC and OpenMP but did not enable offloading support then you will get an error
message that like the following.

```bash
g++ -o omp_offload_test0 -fopenmp -O3 omp_offload_test0.cpp  -fno-stack-protector -foffload=nvptx-none
```
```bash 
lto-wrapper: fatal error: could not find accel/nvptx-none/mkoffload in
/usr/lib/gcc/x86_64-linux-gnu/9/:/usr/lib/gcc/x86_64-linux-gnu/9/:/usr/lib/gcc/x86_64-linux-gnu/:/usr/lib/gcc/x86_64-linux-gnu/9/:/usr/lib/gcc/x86_64-linux-gnu/
(consider using ‘-B’)
```

#### 4. `nvcc` won't compile C++/CUDA file. 

If you have CUDA 10 or older than it will not play with GCC > 8. In such cases, you will get an
error message like the following when you try to build a .cu file with `nvcc`. 

```bash
nvcc hello.cu 
In file included from /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime.h:83,
                 from <command-line>:
/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/host_config.h:129:2: error: #error -- unsupported GNU version! gcc versions later than 8 are not supported!
  129 | #error -- unsupported GNU version! gcc versions later than 8 are not supported!
      |  ^~~~~
```

### References 
 * [How to Install GCC Compiler on Ubuntu 18.04](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/)
 * [Stackoverflow:OpenMP offloading with GCC fails](https://stackoverflow.com/questions/62855136/openmp-offloading-with-gcc-fails-with-ptx-assembly-aborted-due-to-errors) 
 * [Stackoverflow: OpenMP offloading says 'fatal error: could not find
   accel/nvptx-none/mkoffload'](https://stackoverflow.com/questions/62855087/openmp-offloading-says-fatal-error-could-not-find-accel-nvptx-none-mkoffload) 
   






