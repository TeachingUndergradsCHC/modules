## [D1] Heterogeneous Programming with OpenMP
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)


### Description

This module introduces students to GPU offloading with OpenMP. Support for GPU offloading was added
in OpenMP 4.5 with the introduction of the target offload directive. Both LLVM and GCC OpenMP
implementations currently support this mode of task offloading. 

The module first briefly reviews OpenMP. The review is limited to just the core set of pragmas
for loop parallelization. SIMD and SIMT programming models are also reviewed. Following this review,
OpenMP pragmas for task offloading are introduced. The pragmas include target, teams and
distribute. A vector multiplication kernel is used as the driving example. The map clause associated
with task offloading pragmas are also discussed. 

### Context

This module requires that students have some background in parallel programming. As such it is
ideally suited for an upper-level parallel programming course. Sophomore-level Data Structures
courses in which there is some coverage parallel programming, may also be suitable for introducing
this module. 

This module is not meant as a substitute for CUDA or OpenCL programming. It's primary is to get
students up-and-running with programming GPUs quickly. 

### Topics

HC topics covered in this module are listed below. Bloom's classification is shown in brackets

  * Concurrency and parallelism [K]
  * SIMD and SIMT parallelism [K]
  * GPU Acceleration [A]

### Learning Outcomes

Having completed this module, students should be able to 

  * implement simple loop-centric parallel programs that employ OpenMP pragmas to do GPU task offloading 
  


### Instructor Resources

The teaching material included with this module include the following

  * [Slide Deck]
  * [Reference Material]
  * Pedagogical Notes 

All material available for download from the [ToUCH git repository](https://github.com/TeachingUndergradsCHC/modules.git)  


### References 


