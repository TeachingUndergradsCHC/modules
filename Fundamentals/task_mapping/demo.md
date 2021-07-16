## [Demo] Task Mapping on a DVFS-enabled Heterogeneous System
[A2] Task Mapping on Soft Heterogeneous Systems   
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)


### Description 

Demonstrate the performance and energy impact of operational frequency on heterogeneous multicore systems. 

### Software and Tools

The following Linux tools are used in this demo.

  * `cpufrequtils`
  * `cpupower`
  * `perf`
  * `energy`
  * `taskset`

The demo also includes a simple C++/OpenMP code that performance matrix-vector multiplication in
parallel. 


### Environment

Below are instructions for setting a homogenous multicore system as a DVFS-supported heterogeneous platform. 
These steps should be carried out prior to class time. We created a [script](./code/build_hc_env.sh)
to carry out these tasks automatically. Note the below tasks require root access. Follow the
guidelines in the script if root access is not available. 

**0. Download sample codes and utility scripts from the ToUCH repo**

An OpenMP parallel implementation of matrix-vector multiplication is used as a running example for
this demo. There are three utlity scripts for tweaking the frequencies.  

```bash 
git clone https://github.com/TeachingUndergradsCHC/modules.git
```
 
**1. Install necessary packages and their dependencies**

Install `cpufrequtils`


```bash
sudo apt install cpufrequtils
```

Install `perf`, `taskset` and `cpupower` if they are not alreay installed. These tools are available
in the common tools package 


```bash
sudo apt install linux-tools-common
```


```bash
sudo apt install linux-tools-`uname-r`
```

**2. Check CPU clock frequencies**

Clock frequencies of individual cores can be inspected with various utilites. 


```bash
cpufreq-info
```

The ToUCH repository has a script that provides cleaner output. This script might be more suitable for the in-class demo. 


```bash
pwd
```


```bash
./check_clk_speed.sh
```

**3. Lower frequencies for a subset of cores**

We will simulate a less powerful (i.e., _little_) core by lowering its frequency to the lowest allowed
value. To lower the frequency of an individual we can use the `cpupower` utility. We need to root privileges to change the clock frequency (obviously!). The commands below lowers the frequency of core 0 to 1.80 GHz. 


```bash
sudo cpupower -c 0 frequency-set -d 1800000
sudo cpupower -c 0 frequency-set -u 1800000
```

Verify if the change has taken effect


```bash
./check_clk_speed.sh
```

The syntax for the `cpupower` utility is a little cumbersome when we are trying to fix the frequency to a specific value. The `set_clk_speed.sh` script in the ToUCH repo is a wrapper around `cpupower` that provides a cleaner interface. 


```bash
sudo ./set_clk_speed.sh 0-3 2.4
```


```bash
./check_clk_speed.sh
```

There is another script `reset_clk_speed.sh` that resets the frequencies to their default values. 


```bash
./reset_clk_speed.sh 0-15
```


```bash
./check_clk_speed.sh
```

To configure this 16-core system as "big-LITTLE", we will lower the frequencies for cores 0-7 and leave the rest at their defaul values. Cores 0-7 will serve as the _little_ cores and 8-15 will serve as the _big_ cores. Other more complex configurations can be easily set up if the instructor chooses to do a more involved (e.g., in a CS2 course rather CS1)


```bash
./set_clk_speed.sh 0-7 1.8
```


```bash
./check_clk_speed.sh
```

### Instructions 

The main steps for the in-class demo are outlined below

**1. Discuss heterogeneous system.**

Log into system that has been set up to simulate a heterogeneous system (or use this notebook) and review it's attributes.


```bash
cpufreq-info
```

**2. Review matrix-multiply code**

Pull up the matrix-vector source code in an editor and do a walk-through.

  * discuss command-line arguments 
  * discuss basics of an OpenMP directive
  
```C++
double dot_prod(double *x, double *y, int n) {
  double sum = 0.0;
  int i;
#pragma omp parallel for reduction(+:sum)
  for (i = 0; i < n; i++)
      sum += x[i] * y[i];
  return sum;
}

void matrix_vector_mult(double **mat, double *vec, double *result,
                        long long rows, long long cols) {

  /* not parallelelized to ensure runtimes are more meaningful */
  int i;
  for (i = 0; i < rows; i++)
    result[i] = dot_prod(mat[i], vec, cols);
}
```

**3. Build the code on the command-line**


```bash
gcc -o matvec -fopenmp -O3 matvec.c
```

 `matvec` is parallelized with OpenMP. So the `-fopenmp` flag is required. Compiling at `-O3` is
   likely to give more predictable performance numbers. 
   
**4. Run and time the sequential and parallel version of the code**

Run the code with a single thread (i.e., serial version). The matrix size and number of reps can be
adjusted based on the system where the code is running and the amount of time to be devoted to this
demo. With 10000 and 20 the sequential version should run for 3-4 seconds. 


```bash
time ./matvec 10000 20 1
```


```bash
time ./matvec 10000 20 4
```

Discuss the performance improvements with parallelization. Time permitting, the code can be run with
2, 4, ... N threads (where N = number of processing cores on the system) to show the scalability of
the code and discuss Amdahl's Law. 

**4. Discuss mapping of threads to processors**

   Introduce the `taskset` utility and discuss how it can be used to map threads to processing cores.


```bash
## run program on core 0 with 4 threads 
taskset -c 0 ./matvec 10000 20 4
```


```bash
## run program on 2 cores (2 and 5) with 4 threads 
taskset -c 2,5 ./matvec 10000 20 4
```

**5. Run code on _little_ cores**
  
  Run the code on the cores set up as little cores and measure execution time.


```bash
taskset -c 0-7 ./matvec 10000 20 8
```

Re-run the code and measure detailed performance metrics with `perf`


```bash
perf stat taskset -c 0-7 ./matvec 10000 20 8
```

Re-run the code and measure power and energy


```bash
likwid-perfctr -c 0-7 -g ENERGY taskset -c 0-7 ./matvec 10000 20 8
```

**6. Run code on _big_ cores**

   Run the code on the cores set up as little cores and measure execution time.


```bash
cat /proc/cpuinfo
```

Re-run the code and measure power and energy.


```bash
likwid-perfctr -c 8-15 -g ENERGY taskset -c 8-15 ./matvec 10000 20 8
```

**7. Discuss the implications of the results** 

   * little cores will consume less power than big cores
   * little cores will have lower performance than big cores
   * threads must be mapped to cores based on the characteristic of the application and the target
     objective
