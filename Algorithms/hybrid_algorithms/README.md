## [B1] Hybrid Algorithms 
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)

### Description 
This module provides a brief introduction to hybrid algorithms. Specifically, it covers the design and analysis
of hybrid CPU-GPU algorithms in which tasks are performed both on a conventional CPU and an accelerator GPU. 
An accelerator node is introduced in the parallel computation DAG to discuss dependencies in
accelerator offloaded tasks. The dynamic multithreading programming model from CLRS is extended to include the
notion of accelerator offloading via the task keyword. This extended model is used to discuss the complexity of
hybrid parallel divide-and-conquer algorithms. The module concludes with a walk-through of a GPU
accelerated implementation of a sorting algorithm using the OpenMP task directive. 


### Context 

This modules is intended for an undergraduate algorithms course, which is typically a third or
fourth course in most curricula. As such it is expected that students have the following background
when taking this class 
  * basic understanding of parallelism and the shared-memory parallel programming models 
  * basic understanding of a computation DAG and task dependence 
  * complexity analysis of serial algorithms

### Topics 

HC topics covered in this module are listed below. Bloom's classification is shown in brackets

* Accelerator Offloading [K]
* Hybrid Algorithms [C]
* Heterogeneous Parallel Programming Models [K]
* Hybrid parallel Divide-and-Conquer Algorithm [C]
* Complexity analysis of Hybrid parallel algorithms [A]
* Language features for task offloading [K]

### Learning Outcomes

Having completed this module, students should be able to 

 * identify dependencies in a hybrid-parallel computation DAG 
 * analyze complexity of hybrid parallel algorithms using recurrences
 * understand the use of the **target** directive to offload tasks to GPU in OpenMP

### Instructor Resources 

  * [Slide Deck](./lecture_slides.pptx): with instructor annotations
  * [Code]():
  * [Exercises](./exercises.md): in-class exercises, homework problems and exam questions (with solutions)
  * [Reference Material](./reference_material.md): additional resources for instructors
  * Pedagogical Notes: suggestions drawn from author's own experience in teaching this module 



