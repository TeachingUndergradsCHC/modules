## [C2] GPU Memory Hierarchy 
Jacob Newcomb,
Choudry Abdul Rehman,
Justin Douty,
David Bunde [\<dbunde@knox.edu\>](mailto:dbunde@knox.edu)

### Description

This module discusses the GPU memory hierarchy by showing how the
performance of a matrix-multiply code can be improved with tiling,
which aims to improve memory performance.
Rather than a standard triply-nested loop that computes the result
location by location, the tiled algorithm loads submatrices of the
input into shared memory and computes part of the result for an entire
submatrix of the result.

The module is based on an example from a well-known text [[1]](#kirk10).

### Context

The module is intended as second module on GPU programming after
students have been introduced to GPU programming and its SIMD
programming model.
(It could be a successor to our [Introduction to CUDA Programming module](../../Programming/cuda).)
The idea is to make an analogy to CPU caching and reinforce the idea
of caching; GPU shared memory is used as a programmer-controlled cache.
Because of its prerequisites, this module is appropriate for a
mid-level systems course or an upper-level elective.
I plan on using it in an Introduction to Systems course in which both
CUDA programming and caching are introduced.

My students have found it easier to use 
[Google Colab](https://colab.research.google.com) to run the code in
this module than to use ssh to access departmental computing
resources.
Colab provides an interactive computing environment running
[Jupyter](https://jupyter.org/) notebooks with access to GPUs.
As an added wrinkle, the GPU students run on changes (thus, changing
its number of cores and architecture) when they restart
the notebook, which can lead to different results.
See the setup resource below for additional information on using
Google Colab.

### Topics

HC topics covered in this module are listed below. Bloom's
classification is shown in brackets 

  * GPU Acceleration [A]
  * Memory heterogeneity [A]
  * Opimizing memory performance [C]

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
  * [Colab notebook](./cudaMem.ipynb) (open it and then click "open in colab")
  * [Skeleton of tiled matrix multiply](./incomplete_tiled_matrix_mult.cu): 
    A version partially converted to using tiling.
    (A completed version is available to instructors upon request.)
  * [Information on using Google Colab](./colab.md)

All material available for download from the [ToUCH git repository](https://github.com/TeachingUndergradsCHC/modules.git)  

### References

1. <a name="kirk10"></a>D.B. Kirk and W.-m.W. Hwu. Programming massively parallel
processors.  Sections 4.4-4.6, pages 84-96, Morgan Kaufmann, 3rd edition, 2017.


