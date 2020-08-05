### [Demo] Hello World in OpenMP
[D1] Heterogeneous Programming with OpenMP  
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)

#### Objective

An in-class interactive walk-through of the Hello World program parallelized using OpenMP. The
demo covers the following

  * setting up the environment to build and run OpenMP application
  * building OpenMP application with `gcc`
  * `pragma omp parallel`

#### Transcript

1. **Setting up the environment:** Mainstream compilers such as GCC and LLVM provide their own
implementations of OpenMP. The OpenMP libraries and header files are packaged with the compilers. In
general no software packages need to be installed to build and run OpenMP applications, as long as
there is a recent compiler in your system.

    You can check the version of your compiler in the following way 
   
		(ada)% gcc --version
	    gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
	    Copyright (C) 2017 Free Software Foundation, Inc.
	    This is free software; see the source for copying conditions.  There is NO
	    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

		(ada)% clang --version
		clang version 12.0.0 (https://github.com/llvm/llvm-project.git
		041c7b84a4b925476d1e21ed302786033bb6035f) 
		Target: x86_64-unknown-linux-gnu
		Thread model: posix
		InstalledDir: /usr/local/bin


2. **Building and running an OpenMP program:** Below is the canonical Hello World program written in C. 

		#include<stdio.h>
		int main() {
		  printf("Hello World!\n");
		  return 0;
		}

    We will implement an OpenMP version of this program. The first step in writing an OpenMP program
    is including the header file

        #include<omp.h>

    On Linux systems, this file can be found in `/usr/include`. Since this is in the search
    path for headers, there is no need to specify an included path in the compilation command. We can
    compile and create an executable with the following command. 

        gcc -o hello -fopenmp hello.c

    The only difference from a regular build command is that the inclusion of the `-fopenmp`
    flag. This flag tells `gcc` that we are building an OpenMP application. We can now execute this
    program, just like a sequential program. 

        ./hello


3.  **OpenMP pragmas:** OpenMP uses a pragma based syntax. All parallelization and associated directives
    must specified via pragmas or pre-processing directives. All pragmas have the following format
   
        #pragma omp <directive> [ options ]
     
    `#pragma` tells the compiler that this line is to be processed by a pre-processor (not the
    compiler itself). `omp` says that the directive is to be processed by OpenMP. `<directive>` specifies
    the action to be taken on the code that follows immediately. The `<directive>` can be followed by some
    optional arguments (more on this in the next demo). 


4. **The `parallel` pragma:** One of the simplest pragmas in OpenMP is the `parallel` directive. It can
   be used to parallelize a block of code within an application. We will insert the parallel
   directive into our Hello World program. 

		#include<stdio.h>
        #include<omp.h>
		int main() {
		  #pragma omp parallel
		  printf("Hello World!\n");
		  return 0;
		}
	
	This directive will execute the `printf` statement in parallel. Essentially, OpenMP will create
    _n_ threads where each thread will execute the `printf` statement. All _n_ threads will execute
    in parallel. We can build and execute this code as before. *Can we predict the output?*
	
		(ada)%  gcc -o hello -fopenmp hello.c
		(ada)% ./hello 
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World
		Hello World

	*How many threads were created?* We can use the `wc` utility to count the number of lines of
    output from any program. 
	
	    (ada)% ./hello | wc -l 
		12

	OpenMP decided to create 12 threads in this case. *Can we guess why?*. If we do not tell OpenMP
    how many threads to use, it will apply its own judgment to select the number of threads. In many
    situations, the number of threads will correspond to the number of processing cores
    available. We can check the number of cores available with `lscpu` command 
	
		(ada)% lscpu
		Architecture:        x86_64
		CPU op-mode(s):      32-bit, 64-bit
		Byte Order:          Little Endian
		CPU(s):              12
		On-line CPU(s) list: 0-11
		Thread(s) per core:  1
		Core(s) per socket:  6
		Socket(s):           2
		NUMA node(s):        2
		Vendor ID:           GenuineIntel
		CPU family:          6
		Model:               63
		Model name:          Intel(R) Xeon(R) CPU E5-2609 v3 @ 1.90GHz
		Stepping:            2
		CPU MHz:             1198.677

	By default OpenMP pragmas apply to the next statement only. This is often not very useful for
    parallelization as we just saw with our first Hello World example. If we want the `pragma`to
    have an impact on a block of code then we can enclose the region with curly braces `{}` (_almost_
    similar to what we do in C/C++)

		int main() {
		  #pragma omp parallel {
            printf("Hello World\n");
		  }
		  return 0;
		}
	
	The above prompts the compiler to throw the following error. The error message is somewhat
    cryptic. *Can we figure out why the compiler is complaining?*
	
		(ada)%  gcc -o hello -fopenmp hello.c
		hello.c: In function ‘main’:
		hello.c:6:24: error: expected ‘#pragma omp’ clause before ‘{’ token
		#pragma omp parallel {
			                 ^
		hello.c: At top level:
		hello.c:9:3: error: expected identifier or ‘(’ before ‘return’
		return 0;
		^~~~~~
		hello.c:10:1: error: expected identifier or ‘(’ before ‘}’ token
		}
		^

	Unlike C/C++, blocks in OpenMP _must_ start on a newline. Remember, OpenMP directives are being
    processed by the pre-processor, not the compiler. Small sacrifice in style for some of you. We
    can now add multiple statements inside the block to be parallelized by OpenMP

		int main() {
		  #pragma omp parallel 
		  {
            printf("Hello World\n");
            printf("Goodbye World!\n");                                     
		  }
		  return 0;
		}


		(ada)%  gcc -o hello -fopenmp hello.c
		(base) (ada)% ./hello 
		Hello World
		Hello World
		Hello World
		Hello World
		Goodbye World!
		Hello World
		Goodbye World!
		Hello World
		...

	We observe that Hello and Goodbye statements are not being printed in order. OpenMP has created
    12 threads for the block Each threads executes both statements in the block and all threads are running in
    parallel. The output is dependent on which threads gets control over I/O first and will change
    from one run to the next. Of course, in real programs, we will want more control over the
    parallel execution. 

5. **OpenMP API:* OpenMP provides an extensive API to get information from executing
   threads and to configure the parallel execution environment. `omp_set_num_threads()` allows us to
   tell OpenMP how many threads it should in a parallel block of code. `omp_get_num_threads()` gives
   us the number of threads that OpenMP is actually using. This function must be called from inside
   a parallel region. If called from outside it returns 1. Each thread created by OpenMP has a
   unique ID (this is different from the thread ID maintained by the OS). The thread can be
   retrieved at runtime with `omp_get_thread_num()`. 
   
    We will now utilize these functions to better track the parallel execution of our Hello World
    program. 

		#include<stdio.h>
		#include<omp.h>

		int main() {

     	  omp_set_num_threads(4);
		  #pragma omp parallel
          {
            printf("Hello World from thread %u of %u.\n", omp_get_thread_num(), omp_get_num_threads());
            printf("Goodbye World from thread %u of %u.\n", omp_get_thread_num(), omp_get_num_threads());
          }
          return 0;
        }

		(ada)% gcc -o hello -fopenmp hello.c
		(ada)% ./hello 
		Hello World from thread 0 of 4!
		Goodbye World from thread 0 of 4!
		Hello World from thread 1 of 4!
		Goodbye World from thread 1 of 4!
		Hello World from thread 3 of 4!
		Goodbye World from thread 3 of 4!
		Hello World from thread 2 of 4!
		Goodbye World from thread 2 of 4!

    The number of threads in `omp_set_num_threads()` does not have to be a compile-time constant. It
    can be determined at runtime. 
	
	








