### Selecting a GPU Device

If multiple devices are connected to OpenMP then by default the `target` directive will offload the task to the _default_ GPU. Generally, the default GPU is the one with device ID 0. In some cases, we may want to specify on which device we want our code to run. To do this we can combine the `device` clause with the `target` directive.

Consider the following example code that scales the values in a floating-point array


```python
%%writefile scale.cpp
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
  for (unsigned i = 0; i < 1; i++) 
    cout << data[i] << endl;

  return 0;
}
```

    Overwriting scale.cpp


We can offload the scaling computation to one of the available GPUs using the `target` directive


```python
%%writefile scale.cpp
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
  for (unsigned i = 0; i < 1; i++) 
    cout << data[i] << endl;

  return 0;
}
```

    Overwriting scale.cpp


We compile the code with appropriate flags needed for target offloading


```python
!g++ -o scale scale.cpp -fno-stack-protector -foffload=nvptx-none -fopenmp
```

Now let's run the code and time the GPU kernel. Unlike CUDA, since we are not explicitly defining a GPU function, OpenMP will pick it's own name for the kernel. Generally, kernel names will have the form `name_of_function$_omp_$_fn$i` where `function` is the function where the computation is taking place and `$i` is and index maintained by OpenMP. 

`nvprof` will prints a lot of detailed information. Since, for now we are only interested in the GPU kernel execution time, we can extract that information by grepping for the kernel name. Since we have only one GPU task and it resides in `main()`, we can just extract the relevant information by grepping `main`   


```python
!/usr/local/cuda/bin/nvprof  ./scale 2>&1 | grep "main"
```

     GPU activities:   99.88%  12.025ms         1  12.025ms  12.025ms  12.025ms  main$_omp_fn$0


Since we did not specify a device, the offloaded task executed on the default GPU with device id 0. We can check the specs for the available GPU using `nvivia-smi`


```python
!nvidia-smi
```

    Wed Jul 14 10:17:20 2021       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.80       Driver Version: 460.80       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Quadro K620         Off  | 00000000:03:00.0 Off |                  N/A |
    | 34%   43C    P8     1W /  30W |     90MiB /  1993MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla K40c          Off  | 00000000:04:00.0 Off |                    0 |
    | 23%   34C    P8    21W / 235W |      5MiB / 11441MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A      1308      G   /usr/lib/xorg/Xorg                 56MiB |
    |    0   N/A  N/A      1467      G   /usr/bin/gnome-shell               29MiB |
    |    1   N/A  N/A      1308      G   /usr/lib/xorg/Xorg                  4MiB |
    +-----------------------------------------------------------------------------+


On the target system, the default GPU is Tesla K40, which is an older and less powerful GPU based on the Kepler architecture. Running the code on the other GPU based on the Maxwell architecture (still old, but better than Kepler) might give us some performance boost. 

To offload the code to the GPU with device ID 1, we can simply spicify the device ID in the target directive as shown below. 


```python
%%writefile scale.cpp
#include<iostream>
#include<omp.h>

using namespace std;

int main() {
  unsigned N = 10000;
  // float *data = static_cast<float *>(malloc(N * sizeof(float)));
  float data[N];
  for (unsigned i = 0; i < N; i++) 
    data[i] = i;
#pragma omp target device(1) 
  for (unsigned i = 0; i < N; i++) 
    data[i] *= 3.14;
  
  cout << "Computation Done!" << endl; 
  for (unsigned i = 0; i < 1; i++) 
    cout << data[i] << endl;

  return 0;
}
```

    Overwriting scale.cpp



```python
!g++ -o scale scale.cpp -fno-stack-protector -foffload=nvptx-none -fopenmp
```


```python
!/usr/local/cuda/bin/nvprof  ./scale 2>&1 | grep main
```

     GPU activities:   99.84%  8.7503ms         1  8.7503ms  8.7503ms  8.7503ms  main$_omp_fn$0


By selecting the right device for offloading, we were able to get a 50% performance boost. Of course, we haven't parallelized the code yet. So this performance improvement is not really meaningful. 
