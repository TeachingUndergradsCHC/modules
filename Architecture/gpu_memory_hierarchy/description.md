## [C2] GPU Memory Hierarchy 
Jacob Newcomb,
Choudry Abdul Rehman,
David Bunde [\<dbunde@knox.edu\>](mailto:dbunde@knox.edu)

### Description

This module discusses the GPU memory hierarchy by showing how the
performance of a matrix-multiply code can be improved with tiling,
which aims to improve memory performance.
Rather than a standard triply-nested loop that computes the result
location by location, the tiled algorithm loads submatrices of the
input into shared memory and computes part of the result for an entire
submatrix of the result.

### Context

The module is intended as second module on GPU programming after
students have been introduced to GPU programming and its SIMD
programming model.
The idea is to make an analogy to CPU caching and reinforce the idea
of caching; GPU shared memory is used as a programmer-controlled cache.
Because of its prerequisites, this module is appropriate for a
mid-level systems course or an upper-level elective.
I plan on using it in an Introduction to Systems course in which both
CUDA programming and caching are introduced.

### Topics

HC topics covered in this module are listed below. Bloom's
classification is shown in brackets 

  * GPU Acceleration [K]
  * Memory heterogeneity [K]

### Learning Outcomes

Having completed this module, students should be able to 

  * Explain the properties and limitations of shared memory in CUDA
    programming
  * Write code using GPU shared memory in CUDA 
  * Estimate the number of memory operations for simple programs

### Instructor Resources

This module includes the following teaching materials:

  * Slides ([.pptx](./lecture_slides.pptx), [.pdf](./lecture_slides.pdf))
  * [(Untiled) CUDA code for matrix multiply](./matrix_multiply.cu):
    An implementation that doesn't use tiling
  * [Skeleton of tiled matrix multiply](./incomplete_tiled_matrix_mult.cu): 
    A version partially converted to using tiling

All material available for download from the [ToUCH git repository](https://github.com/TeachingUndergradsCHC/modules.git)  



