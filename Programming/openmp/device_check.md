### OpenMP Device Check 

OpenMP uses host-device programming model. Multiple devices are connected to a host. Initial thread begins execution on the host. Tasks are offloaded to devices. In this model a GPU is a _device_. For GPU `target` offloading to work, we need to make sure that at least one GPU is connected to OpenMP. The following program checks and reports the number of devices connected to OpenMP. 

The cell below with can be edited in-place. When executed, the cell will save the file `device_check.cpp` in the current directory. 


```python
%%writefile device_check.cpp
#include<stdio.h>
#include<omp.h>

int main() {
  unsigned int devs = omp_get_num_devices();

  if (devs > 0) 
    printf("There are %d devices\n", omp_get_num_devices());
  else
    printf("No devices connected. OpenMP GPU offloading will not work.\n"); 
    return 0;
}
```

    Overwriting device_check.cpp


We can compile the code with the following


```python
!g++ -o device_check device_check.cpp -fno-stack-protector -foffload=nvptx-none -fopenmp
```

Note, in this instance the `-fno-stack-protector` and `-foffload=nvptx-none` flags are not necessary since we are not offloading anything to the GPU. Using the flags doesn't hurt in anayway and we include them here just as a matter of practice.  

Now let's run the code.


```python
!./device_check
```

    There are 2 devices


In general, the number of devices should match the number of GPUs on your system/compute node. If the program shows no devices are connected then OpenMP target offload enviroment has not been set up properly.


```python

```
