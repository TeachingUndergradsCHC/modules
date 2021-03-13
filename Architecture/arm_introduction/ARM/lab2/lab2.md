## Lab 2: ARM vs. Thumb speed/codesize tradeoffs

### Description
The ARM has several features and additional processing elements 
that require the software developer to decide how and on which
processing element code will run.  One of those decisions requires an understanding of
ARM and Thumb mode.  In this lab students will build and run code
targeting different sub-architectures: ARM, Thumb 1, and Thumb 2.

There are two simple pieces of benchmark code, one that performs the Sieve of
Eratosthenese algorihm and another that performs matrix multiply.  You will use some
simple command line tools to try to
understand  the codesize vs speed tradeoffs between the various
ARM sub-ISA's

### Environment

You will be building and running code from the Raspbian (Linux)
command line on your Raspberry Pi.

### Tools

Familiarize yourself with the following tools. 

  * `readelf`
  * `gdb`
  * `make`
  * `size`

### Instructions

#### 1. Download the lab 

Download the lab2.zip file and unzip it:

```
unzip lab2.zip
```
	
This zipfile includes several C files, `sieve.c`, `sieve-main.c`, `mult.c`, mult-main.c`, and a `makefile`
	
#### 2. Run the makefile

You must have the GNU toolchain installed.  Simply enter the directory
where you unzipped the files and type `make`.

#### 3. Conduct Performance Experiments

Each of the two sample benchmarks are compiled in three different
ways:  ARM mode, Thumb1 mode, and Thumb2 mode.  A total of six
executables are built: `mult-arm`, `mult-thumb1`, `mult-thumb2`, `sieve-arm`,
`sieve-thumb1`, and `sieve-thumb2`.  You should run each of them.  Each executable
will print out a series of numbers indicating the wall-clock time
for running the benchmark.  You should record these values and take an average.

#### 4. Check the codesize of the executables

We want to measure only the codesize of the code we are benchmarking,
not the test test harness code.  Try the following command:

```
elfread -a sieve-arm.o
```
Can you find the codesize in the output for the function `sievemark()`?

An easier way to get the size of the executable file is:
`size sieve-arm.o`
Now the code size should be very obvious. Since this object file contains
only one function, the reported code size is the code size (in bytes) of that function.
Find the codesize for each
of the six benchmarks by running the `size` command on sieve-arm.o,
sieve-thumb1.o, sieve-thumb2.o, mult-arm.o, mult-thumb1.o, and
mult-thumb2.o

#### 5. Examine the assembly files


#### 6. Analyze the data

* Describe the difference in performance observed in Step 3 
* Describe the differences in code-size observed in Step 4
* Investigate the system call time().  Why might it not be the best
method to compare running times of different programs?
* What observations can be made about the effectiveness of Thumb2?
