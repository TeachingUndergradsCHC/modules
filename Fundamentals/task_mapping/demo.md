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

**1. Install necessary packages and their dependencies**

Install `cpufrequtils`
  
     sudo apt install cpufrequtils

Install `perf`, `taskset` and `cpupower` if they are not alreay installed. These tools are available
in the common tools package 

     sudo apt install linux-tools-common
     sudo apt install linux-tools-4.15.0-47-generic
     

**2. Lower frequencies for a subset of cores**

We simulate a less powerful (i.e., _little_) core by lowering its frequency to the lowest allowed
value. In this example we select cores 0 and 1 to be the _little_ cores. To lower the frequencies for
cores 0 and 1

     cpupower -c 0 frequency-set -d 1800000
     cpupower -c 0 frequency-set -u 1800000
     cpupower -c 1 frequency-set -d 1800000
     cpupower -c 1 frequency-set -u 1800000


The above will set up the multicore system with two _big_ cores (cores 2 and 3) and two _little_
cores (cores 0 and 1). Other more complex configurations can be easily set up if the instructor
chooses to do a more involved (e.g., in a CS2 course rather CS1)

**3. Download example code from the ToUCH repo**

An OpenMP parallel implementation of matrix-vector multiplication is used as a running example for
this demo. The code is found here. 


### Instructions 

The main steps for the in-class demo are outlined below

**1. Discuss heterogeneous system.**

Log into system that has been set up to simulate a heterogeneous system and review it's attributes. 

    cat /proc/cpuinfo
    cpufreq-info
	
**2. Review matrix-multiply code**

Pull up the matrix-vector source code in an editor and do a walk-through.

  * discuss command-line arguments 
  * discuss basics of an OpenMP directive


**3. Build the code on the command-line**

    gcc -o matvec -fopenmp -O3 matvec.c

   `matvec` is parallelized with OpenMP. So the `-fopenmp` flag is required. Compiling at `-O3` is
   likely to give more predictable performance numbers. 
   
**4. Run and time the sequential and parallel version of the code**

Run the code with a single thread (i.e., serial version). The matrix size and number of reps can be
adjusted based on the system where the code is running and the amount of time to be devoted to this
demo. With 10000 and 20 the sequential version should run for 3-4 seconds. 

    time ./matvec 10000 20 1
	

Run the code with 2 threads and time the run. 

    time ./matvec 10000 20 2

Discuss the performance improvements with parallelization. Time permitting, the code can be run with
2, 4, ... N threads (where N = number of processing cores on the system) to show the scalability of
the code and discuss Amdahl's Law. 

**4. Discuss mapping of threads to processors**

   Introduce the `taskset` utility and discuss how it can be used to map threads to processing cores.

	## run program on core 0 with 4 threads 
    taskset -c 0 ./matvec 10000 20 4
	## run program on 2 cores (2 and 5) with 4 threads 
    taskset -c 2,5 ./matvec 10000 20 4

Discuss software and hardware threads and impact on performance. 

**5. Run code on _little_ cores**
  
  Run the code on the cores set up as little cores and measure execution time. 

    taskset -c 0-3 ./matvec 10000 20 4

  Re-run the code and measure detailed performance metrics with perf

    perf stat taskset -c 0-3 ./matvec 10000 20 4

  Re-run the code and measure power and energy

    likwid-perfctr -c 0-3 -g ENERGY taskset -c 0-3 ./matvec 10000 20 4

**6. Run code on _big_ cores**

   Run the code on the cores set up as little cores and measure execution time.

    time taskset -c 4-7 ./matvec 10000 20 4

   Re-run the code and measure power and energy

    likwid-perfctr -c 4-7 -g ENERGY taskset -c 4-7 ./matvec 10000 20 4


**7. Discuss the implications of the results** 

   * little cores will consume less power than big cores
   * little cores will have lower performance than big cores
   * threads must be mapped to cores based on the characteristic of the application and the target
     objective




