## [D1] Heterogeneous Programming with OpenMP
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)


### Description

This module covers GPU task offloading in OpenMP. Support for GPU offloading was added
in OpenMP 4.5 with the introduction of the `target` offload directive. Both LLVM and GCC OpenMP
implementations currently support this mode of task offloading. 

The module first briefly reviews OpenMP. The review is limited to only a core set of pragmas
for loop parallelization. SIMD and SIMT models of parallelism, as it applies to GPU computing, is
also discussed. Following this review, OpenMP pragmas for task offloading are introduced. The
pragmas include `target`, `teams` and `distribute`. A vector multiplication kernel is used as the
driving example. The map clause associated with task offloading pragmas is also discussed.

### Context

This module requires that students have some background in parallel programming. As such it is
ideally suited for an upper-level parallel programming course. Sophomore-level Data Structures
courses in which there is some coverage parallel programming, may also be suitable for introducing
this module. 

This module is not meant as a substitute for CUDA or OpenCL programming. It's primary goal is to get
students up-and-running with programming GPUs quickly. 

### Topics

HC topics covered in this module are listed below. Bloom's classification is shown in brackets

  * Concurrency and parallelism [K]
  * SIMD and SIMT parallelism [K]
  * GPU Acceleration [A]
  * Shared memory programming (OpenMP) [A]
  
### Learning Outcomes

Having completed this module, students should be able to 

  * implement simple loop-centric parallel programs that employ OpenMP pragmas to do GPU task offloading 
  

### Instructor Resources

The teaching material included with this module include the following

  * [Slide Deck](./lecture_slides.pptx): includes instructor annotations
  * [In-class Demo]:  
       * [Hello World in OpenMP](./demo_hello_world.md)
  * Pedagogical Notes: suggestions drawn from author's own experience in teaching this module 
  * [Reference Material](./reference_material.md): additional resources for instructors

All material available for download from the [ToUCH git repository](https://github.com/TeachingUndergradsCHC/modules.git)  


### References 


