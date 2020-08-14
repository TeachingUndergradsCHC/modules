## Lab: ARM vs. Thumb speed/codesize tradeoffs

### Description
The ARM has several features and additional processing elements 
that require the software developer to decide how and on which
PE code will run.  One of those decisions requires an understanding of
ARM and Thumb mode.  In this lab students will build and run code
targeting different sub-architectures: ARM, Thumb 1, and Thumb 2.

You will be given several small pieces of code and a makefile to build
them.  You will use some simple command line tools to try to
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

### Instructions

#### 1. Download all the lab code 

Download the lab1.zip file and unzip it:
	unzip lab1.zip
This includes several C files, sieve.c, sieve-main.c, mult.c, and
	mult-main.c, and a makefile
	
#### 2. Run the makefile

You must have the GNU toolchain installed.  Simply enter the directory
where you unzipped the files and type `make`.

#### 3. Conduct Performance Experiments

Each of the two sample benchmarks are compiled in three different
ways:  ARM mode, Thumb1 mode, and Thumb2 mode.  A total of six
executables are built: mult-arm, mult-thumb1, mult-thumb2, sieve-arm,
sieve-thumb1, and sieve-thumb2.  You should run each of them in the
following manner:
`<executable> `

#### 4. Check the codesize of the executables


#### 5. Examine assembly files


#### 6. Analyze the data

* Describe the difference in performance observed in Step 3 
* Describe the differences in code-size observed in Step 4
* Investigate the system call time().  Why might it not be the best
method to compare running times of different programs?
* What observations can be made about the effectiveness of Thumb2?
