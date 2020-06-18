## [A2] Task Mapping on Soft Heterogeneous Systems 
Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)


### Description

This module covers the concepts of task mapping and scheduling on single-ISA heterogeneous computing
systems. The first part of the module briefly reviews the notions of concurrency, parallelism, and energy
efficiency. This serves as a lead-in to the discussion on the motivation behind the shift towards
heterogeneous processing. Different forms of heterogeneity are introduced including CPU-GPU
heterogeneous execution and heterogeneity that arises in single-ISA systems. 

The second part of the module focuses on task mapping and scheduling on systems that embody soft
heterogeneity; systems in which differences in core compute capabilities are observed as a result of
differences in operating core frequency. To motivate the need for such systems, workload
heterogeneity is discussed with examples from mobile computing. The concepts of task 
mapping and scheduling on sequential and parallel systems are reviewed which sets the stage for the
task mapping problem on soft heterogeneous systems. An in-class interactive demo on a real system
illustrates the performance and energy efficiency challenges of task mapping which factors in operating
frequency. The module is accompanied by a hands-on lab that reinforces these ideas. 


### Context

This module is primarily intended for CS2 students. Although the module introduces parallel
computing concepts before moving on to processor heterogeneity, it is ideally suited for a course with some
coverage of parallel computing material. For example, a course that incorporates a PDC module from
[[1]](#csinparallel),[[2]](#cder), or [[3]](#tues). In the absence of PDC coverage, the length
of this module will need to be increased or it will need to be combined with a PDC module.

This module can be combined with [Module A] Heterogeneous Computing: Elementary Notions in which
case it can potentially be introduced in a CS1 class. For example, a CS1 class designed for Honors
students or one which provides a breadth-first introduction to computer science. 

### Topics

The HC topics covered in this module are listed below. Bloom's classification is shown in brackets

  * Single-ISA Heterogeneity [K]
  * System-on-Chip [K]
  * DVFS and Soft Heterogeneity [K]
  * Energy Efficiency [K]
  * Heterogeneous Tasks and Workloads [K]
  * Task Mapping and Scheduling [C]
  * Tools for thread affinity and CPU frequency scaling [A]
  
### Learning Outcomes

Having completed this module, students should be able to 

  * understand the motivation behind the design of heterogeneous computing systems
  * recognize the importance of energy efficiency on current computing systems
  * understand that tasks in a workload have different demands for compute and memory resources
  * understand the notion of task mapping as performed by an operating system
  * analyze the performance and energy effects of task mapping on a heterogeneous system

### Instructor Resources

The following teaching material are included in the module.
  * [Slide Deck](./lecture_slides.pptx): includes instructor annotations
  * [Lab](./lab.md) : The module includes a lab that provides students hands-on experience in running
    application on a heterogeneous system. The lab will also reinforce the performance and energy
    implications covered in the lecture. The lab includes detailed instructions for the instructor
    in setting up a heterogeneous system on which students will conduct performance experiments. The
    lab requires the student to have some basic familiarity with a Linux environment.   
  * [In-class Demo](./demo.md): includes instructions for setting up a heterogeneous environment within a
    homogeneous multicore system and step-by-step guidelines for running the demo in class  
  * [Reference Material](./reference_material.md): additional resources for instructors
  * Pedagogical Notes: suggestions drawn from author's own experience in teaching this module 

All material available for download from the [ToUCH git
repository](https://github.com/TeachingUndergradsCHC/modules.git)  


### References 

* <a name="csinparallel"></a>CSinParallel Project. <http://csinparallel.org/>
* <a name="cder"></a>Center for parallel and distributed computing curriculum development and
  educational resources 
  (CDER). <http://www.cs.gsu.edu/~tcpp>
* <a name="tues"></a>Parallel Computing in the Undergraduate Curriculum : the Early-and-Often
  Approach. <https://tues.cs.txstate.edu>

