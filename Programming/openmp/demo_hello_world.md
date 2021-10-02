## [Demo] Hello World in OpenMP 
[D2] Heterogeneous Programming with OpenMP  
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)

### Description

An in-class interactive walk-through of the Hello World program, parallelized using OpenMP. 


### Outline 

  * [Setting up OpenMP in Linux](#env)
  * [Compiling and running an OpenMP program](#compile) (`gcc` command-line)
  * [OpenMP pragmas:](#pragma)  `parallel`
  * [OpenMP API:](#api) `omp_set_num_threads(), omp_get_num_threads(), omp_get_thread_num()`


### <a name="env"></a>Setting up OpenMP in Linux
All mainstream compilers today provide integrated support for OpenMP. Each compiler has its own
implementation of the OpenMP standard. The OpenMP libraries and header files are packaged and
distributed with the compiler. So, no software packages need to be installed to build and run OpenMP
applications as long as there is a more-or-less recent compiler installed on the system. 

We can check the version of the compiler in our system as follows (ada is the name of the
machine where the commands in this demo were run). GCC ... 

```
(ada)% gcc --version
gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

... and LLVM 

```
(ada)% clang --version
clang version 12.0.0 (https://github.com/llvm/llvm-project.git 041c7b84a4b925476d1e21ed302786033bb6035f) 
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/local/bin
```

### <a name="compile"></a>Compiling and running an OpenMP program

Below is the canonical Hello World program written in C.
```C
#include<stdio.h>
int main() {
  printf("Hello World!\n");
  return 0;
}
```

We will implement an OpenMP version of this program. Generally, the first step in writing an OpenMP
program is including the header file (although in this trivial example we could have done without this)

    #include<omp.h>

On Linux systems, `omp.h` is located `/usr/include`. Since this is in the compiler's search path for
header files there is no need to specify an include path (with the `-I` flag) in the compilation
command. We can compile and create an executable with the following command.  

    (ada)% gcc -o hello -fopenmp hello.c

The only difference from a regular build command is the inclusion of the `-fopenmp` flag. This flag
tells `gcc` that we are building an OpenMP application. We can now execute this  program from the
command-line just like a serial program. 

    (ada)% ./hello

### <a name="pragma"></a>OpenMP pragmas

OpenMP uses a pragma-based syntax. All parallelization and associated directives must be specified
via pragmas. All pragmas have the following format 
    
    #pragma omp <directive> [ options ]
     
`#pragma` tells the compiler that this line is to be processed by a pre-processor (not the compiler
    itself). `omp` says that the directive is to be processed by OpenMP. `<directive>` specifies the
    action to be taken on the code that immediately follows the pragma. The `<directive>` can be
    followed by a set of optional arguments. In OpenMP terminology, these arguments are called
    *clauses* (more on this in the next demo).  


**The `parallel` pragma:** One of the simplest pragmas in OpenMP is the `parallel` directive. It can
   be used to parallelize a block of code within an application. We will insert the parallel
   directive into our Hello World program. 

```C
#include<stdio.h>
#include<omp.h>
int main() {
  #pragma omp parallel
  printf("Hello World!\n");
  return 0;
}
```

This directive will execute the `printf` statement in parallel. This means that OpenMP will create
    _n_ threads where each thread will execute an instance of the `printf` statement. All _n_
    threads will execute this statement in parallel. We can build and execute this code as before.

_Can we predict the output?_
	
```
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
```

_How many threads were created?_
	
**Thread count:** We can use the `wc` utility to count the number of lines of output from any program. 
	
```
(ada)%./hello | wc -l 
12
```
	
OpenMP decided to create 12 threads in this case. 
	
_Can we guess why?_ 
	
If we do not tell OpenMP how many threads to use, it will apply its own judgment to select the
number of threads. In many situations, the number of threads will correspond to the number of processing cores  available. We can check the number of cores available on our with `lscpu` command 
	
```		
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
```
	
**pragma scope** 

By default OpenMP pragmas apply to the next statement only. This is often
    not very useful for parallelization as we just saw with our first Hello World example. If we
    want the `pragma` to have an impact on a block of code then we can enclose the region with curly
    braces `{}` (_almost_ similar to what we do in C/C++)

```C
int main() {
  #pragma omp parallel {
    printf("Hello World\n");
  }
  return 0;
}
```

The above prompts the compiler to throw the following error. The error message is somewhat cryptic. 
	
_Can we figure out why the compiler is complaining?_
	
	
```
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
```

**Compiler quirks:** Unlike C/C++, blocks in OpenMP _must_ start on a newline. Remember, OpenMP
    directives are being processed by the pre-processor, not the compiler. (Not being able to place
    the opening brace on the same line is a small sacrifice in style for some of us). We can now add
    multiple statements inside the block to be parallelized by OpenMP. 

```C
int main() {
  #pragma omp parallel 
  {
    printf("Hello World\n");
    printf("Goodbye World!\n");
  }
  return 0;
}
```

Let's compile this version. 
 
    (ada)%  gcc -o hello -fopenmp hello.c
	(ada)% ./hello 
	Hello World
	Hello World
	Hello World
	Hello World
	Goodbye World!
	Hello World
	Goodbye World!
	Hello World
	...

**`parallel` semantics:**  We observe that Hello and Goodbye statements are not being printed in
    order. OpenMP has created 12 threads for the block Each threads executes both statements in the
    block and all threads are running in parallel. The output is dependent on which threads gets
    control over I/O first and will change from one run to the next. Of course, in real programs, we
    will want more control over the parallel execution. 

### <a name="api"></a>OpenMP API
OpenMP provides an extensive API to get information from executing threads and to configure the
   parallel execution environment. `omp_set_num_threads()` allows us to 
   tell OpenMP how many threads it should in a parallel block of code. `omp_get_num_threads()` gives
   us the number of threads that OpenMP is actually using. This function must be called from inside
   a parallel region. If called from outside it returns 1. Each thread created by OpenMP has a
   unique ID (this is different from the thread ID maintained by the OS). The thread ID can be
   retrieved at runtime with `omp_get_thread_num()`. 
   
   We will now utilize these functions to track the parallel execution of our Hello World program. 

```C
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
```
	   
This program limits the number of OpenMP threads to 4. The parallel segment then prints out the ID
		of each thread created by OpenMP and the total number of threads. 
 	
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


The number of threads in `omp_set_num_threads()` does not have to be a compile-time constant. It can be determined at runtime. 
	
	








