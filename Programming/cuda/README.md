## [D1] Introduction to CUDA Programming
David Bunde [\<dbunde@knox.edu\>](mailto:dbunde@knox.edu)

### Description

This module introduces CUDA programming using an image processing application. 
It begins with a lecture that introduces General Purpose Graphics
Processing Unit (GPU) programming and CUDA specifically.
The image processing application is then given as a lab or homework assignment.

### Context

This module is intended as a first introduction to GPU programming.
Students are shown some basic examples ("Hello world" and adding
vectors) before tackling the image processing application.
I gave the later as a lab assignment, though it could also be phrased
as a homework assignment.
For courses that are going deeper into CUDA, this module could be
followed by our [GPU Memory Hierarchy module](../../Architecture/gpu_memory_hierarchy).
The given code is in C.
The architectural aspects of CUDA are treated very lightly.
I am using this module in my Introduction to Systems course, which has
parallelism as a major topic.

My students have found it easier to use 
[Google Colab](https://colab.research.google.com) to run the code in
this module than to use ssh to access departmental computing
resources.
Colab provides an interactive computing environment running
[Jupyter](https://jupyter.org/) notebooks with access to GPUs.
Using GPUs does require installing the nvcc compiler.
As an added wrinkle, the GPU students run on changes (thus, changing
its number of cores and architecture) when they restart
the notebook, which can lead to different results.
See the setup resource below for additional information on using
Google Colab.

I typically teach using the Peer Instruction pedagogy, in which
lectures feature multiple choice questions that the students answer
and discuss.
(See [[1]](#pi4cs) for more information.)
Because of this, my slides have some multiple choice questions.
Since not everyone teaches this way, the module includes 2 sets of
slides, one with questions and one for a more conventional
lecture class.
(Even the slides with questions only have a couple of questions at the
 end.
 Thus, they don't fully implement the Peer Instruction pedagogy;
 something else to work on...)

### Topics

HC topics covered in this module are listed below. Bloom's classification is shown in brackets

  * GPU Acceleration [A]
  * SIMD parallelism [A]

### Learning Outcomes

Having completed this module, students should be able to 

  * Write simple programs in CUDA
  * Translate between 1D and 2D coordinates by arranging the 2D points
    in row-major order

### Instructor Resources

The teaching material included with this module are the following:

  * Slides with embedded questions
    ([.pptx](./lecture_slides_pi.pptx), [.pdf](./lecture_slides_pi.pdf))
  * Slides for a traditional lecture
    ([.pptx](./lecture_slides.pptx), [.pdf](./lecture_slides.pdf))
  * CUDA programs discussed during lecture:
    [hello.cu](./hello.cu), [addVectors.cu](./addVectors.cu)
  * [Information on using Google Colab](../../Architecture/gpu_memory_hierarchy/colab.md)
  * Files for lab:
    * [Colab notebook](./cudaBlur.ipynb)  (open it and then click
    "open in colab")
    * Lab handout: [.pdf](./lab.pdf)
    * [ppmFile.h](./ppmFile.h), [ppmFile.c](./ppmFile.c): library
    files for dealing with ppm files
    * [640x426.ppm](./640x426.ppm): sample image file (can also use your own)
    * [noRed.cu](./noRed.cu): kernel that removes red from the image

All material available for download from the [ToUCH git repository](https://github.com/TeachingUndergradsCHC/modules.git)  

### References 

1. <a name="pi4cs"></a>Peer Instruction for CS. <http://peerinstruction4cs.com/>
