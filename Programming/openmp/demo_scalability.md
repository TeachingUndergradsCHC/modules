## [Demo] OpenMP Performance Scalability 
[D2] Heterogeneous Programming with OpenMP  
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)

### Description

This interactive demo discusses the importance and significance of thread count in OpenMP
applications. It is assumed that students know how to write a Hello World program in OpenMP (e.g.,
have completed the ([Hello World in OpenMP](demo_hello_world.md)) tutorial. 

The demo also introduces the `parallel for` directive. A simple matrix-scalar multiplication code is
used as a running example. 

### Outline 

   * [The OpenMP Hello World Program](#hello)
   * [Dynamic Thread Count](#timing)
   * [Parallelizing with `parallel for`](#pragma) 
   * [Thread Count ans Scalability](#thread_count)


### <a name="hello"></a>The OpenMP Hello World Program

Below is the Hello World program with OpenMP parallelization that we wrote in our previous tutorial
([Hello World in OpenMP](demo_hello_world.md)). 

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

The above program will run with 4 OpenMP threads. The parallel segment will print out the ID of each
thread created by OpenMP and the total number of threads launched. 
	 	
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


### <a name="timing"></a>Dynamic Thread Count 

Because we specified the thread count at compile-time, the above program will _always_  launched
with 4 threads. threads. Generally, this is not a good approach when programming with OpenMP. We
want better control over OpenMP threads and be able to specify the number of threads considering a
variety of factors including the target CPU, input size and task granularity. 


To set the number of threads dynamically, we can pass the thread count to the program as a
command-line argument. 

```C
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int main(int argc, char* argv[]) {

  int num_threads;
  if (argc <= 1)
    num_threads = 1;
  else
    num_threads = atoi(argv[1]);

  omp_set_num_threads(num_threads);
  ...
  ...
```

This approach will allow us to choose a different thread count for different runs of the
program. Further, if we want to run the serial version of the code (say, for testing purposes), we
can just pass 1 as the command-line argument. 

Let us run the sequential version and time the run. 

    (knuth)% time ./hello 1
    Hello World from 0!
    Goodbye World from 0!

    real	0m0.004s
    user	0m0.001s
    sys	  0m0.004s

The Linux `time` command doesn't really give us satisfactory resolution for measuring the
performance of this _tiny_ program. We can use
[`perf`](https://perf.wiki.kernel.org/index.php/Main_Page) to get better measurements. 

	(knuth)% perf stat ./hello 1
    Hello World from 0!
    Goodbye World from 0!

    Performance counter stats for './hello 1':

          2.240399      task-clock (msec)         #    0.864 CPUs utilized          
                 0      context-switches          #    0.000 K/sec                  
                 0      cpu-migrations            #    0.000 K/sec                  
               118      page-faults               #    0.053 M/sec                  
         3,788,857      cycles                    #    1.691 GHz                    
         2,289,566      stalled-cycles-frontend   #   60.43% frontend cycles idle   
         1,618,024      stalled-cycles-backend    #   42.70% backend cycles idle    
         3,607,090      instructions              #    0.95  insn per cycle         
                                                  #    0.63  stalled cycles per insn
           628,934      branches                  #  280.724 M/sec                  
            19,700      branch-misses             #    3.13% of all branches        

       0.002592641 seconds time elapsed

Now, let's run the code with 2 threads. 

    (knuth)% perf stat ./hello 2
    Hello World from 0!
    Goodbye World from 0!
    Hello World from 1!
    Goodbye World from 1!

    Performance counter stats for './hello 2':

          2.384980      task-clock (msec)         #    0.912 CPUs utilized          
                 1      context-switches          #    0.419 K/sec                  
                 0      cpu-migrations            #    0.000 K/sec                  
               121      page-faults               #    0.051 M/sec                  
         4,249,512      cycles                    #    1.782 GHz                    
         2,662,123      stalled-cycles-frontend   #   62.65% frontend cycles idle   
         1,932,917      stalled-cycles-backend    #   45.49% backend cycles idle    
         3,694,681      instructions              #    0.87  insn per cycle         
                                                  #    0.72  stalled cycles per insn
           654,539      branches                  #  274.442 M/sec                  
            21,580      branch-misses             #    3.30% of all branches        

       0.002616198 seconds time elapsed

_How much performance improvement do we get by running this code in parallel?_

None! This very simple code is not useful for doing any kind of performance analysis. 

### <a name="pragma"></a>Parallelizing with `parallel for`

Let's look at a code that is slightly more complex. 

```C
for p(j = 0; j < M; j++)
  for (i = 0; i < M; i++)
    b[i][j] = i + j;

t0 = mysecond();
#pragma omp parallel for
  for (int k = 0; k < REPS; k++) {
    for (int j = 0; j < M; j++)
      for (int i = 0; i < M; i++)
        a[i][j] = b[i][j] * 17;
  }

t0 = (mysecond() - t0) * 1.e3;
printf("parallel loop = %3.2f ms\n", t0);
```
The above program scales the values in an array by a constant factor. The loop is parallelized with the
`parallel for` directive. This directive is an extension of the `parallel` directive and is applied
exclusively to the *next* for loop. The `parallel for` directive will equally divide the iterations
of the loop and run them in parallel. The number of threads to be created is passed via a command-line
argument. There's a built-in timer to record the execution time of the parallel loop. 



### <a name="thread_count"></a>Thread Count and Scalability 

Let's build and execute the sequential version of the code. 

```
(knuth)% g++ -o scale scale.c -fopenmp
(knuth)% ./scale 1000 1
result = 578.00
parallel loop = 1936.35 ms
```

Let's run it with 2 threads. 

```
(knuth)% ./scale 1000 2
result = 578.00
parallel loop = 1251.09 ms
```

The parallel version runs significantly faster. However note, even with this very simple code we are
not able to double the performance when we increase the number of threads from 1 to 2. 

_Why?_

See [Heterogeneous Computing: Elementary Notions](../../Fundamentals/elementary_notions) for one
explanation. 


Now let's run the code with 12 threads which is what OpenMP would pick for this system if we did not
specify the thread count ourselves. 

```
(knuth)% ./scale 1000 12
result = 578.00
parallel loop = 419.77 ms
(knuth)% 
```

This gives use about a 5x performance improvement over the sequential code. Not bad... but
not ideal either. 

_What if we kept on increasing the number of threads, do we expect to get more parallelism?_

```
(knuth)% ./scale 1000 32
result = 578.00
parallel loop = 373.80 ms
(knuth)% ./scale 1000 64
result = 578.00
parallel loop = 374.94 ms
(knuth)% ./scale 1000 128
result = 578.00
parallel loop = 375.71 ms
```

_Does this performance pattern reminds us of something?_

This program becomes [compute-bound](https://en.wikipedia.org/wiki/CPU-bound) when the number of
threads is substantially higher than the available processing cores. At that point increasing the
number of threads doesn't give us any benefits (in fact in some cases it can actually hurt due to
thread creation overhead). 

The ideal number of threads for a given program depends on many factors. Often some fine-tuning is
necessary. 

### Exercise 

Compile and run the `matrix-scale` code on your own machine with increasing number of
threads. What is the ideal thread count?
