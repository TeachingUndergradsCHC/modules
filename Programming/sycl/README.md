## [D3] Introduction to SYCL Programming

Kartikay Bhuchar, Amar Puri, Abhishek Thapa, Benson Muite and
David Bunde [\<dbunde@knox.edu\>](mailto:dbunde@knox.edu)

### Description

This module introduces SYCL programming using an image processing application. 
It is a translation of the CUDA code in the module "Introduction to CUDA
Programming". It begins with a lecture that introduces 
General Purpose Graphics Processing Unit (GPU) programming and SYCL specifically.
SYCL code can also be run on a Central Processing Unit (CPU). The image processing 
application is then given as a lab or homework assignment.

### Context

This module is intended as a first introduction to GPU programming.
Students are shown some basic examples ("Hello world" and adding
vectors) before tackling the image processing application.
I gave the later as a lab assignment, though it could also be phrased
as a homework assignment.

The given code is in C and C++.  The architectural aspects of SYCL are 
treated very lightly.  I am using this module in my Introduction to 
Systems course, which has parallelism as a major topic.

My students have found it easier to use 
[Google Colab](https://colab.research.google.com) to run the code in
this module than to use ssh to access departmental computing
resources.
Colab provides an interactive computing environment running
[Jupyter](https://jupyter.org/) notebooks with access to GPUs.
Using GPUs does require installing a SYCL compiler, for portability
[hipSYCL](https://github.com/illuhad/hipSYCL) is used.
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

  * Write simple programs in SYCL
  * Translate between 1D and 2D coordinates by arranging the 2D points
    in row-major order

### Instructor Resources

The teaching material included with this module are the following:

  * Slides with embedded questions
    ([.pptx](./lecture_slides_pi.pptx), [.pdf](./lecture_slides_pi.pdf))
  * Slides for a traditional lecture
    ([.pptx](./lecture_slides.pptx), [.pdf](./lecture_slides.pdf))
  * SYCL programs discussed during lecture:
    [hello.cpp](./hello.cpp), [addVectors.cpp](./addVectors.cpp)
  * [Information on using Google Colab](../../Architecture/gpu_memory_hierarchy/colab.md) 
    (The instructions are largely duplicated in the Colab notebook
     below.)
  * Files for lab:
    * [Colab notebook](./syclBlur.ipynb)  (open it and then click
    "open in colab")
    * Lab handout: [.pdf](./lab.pdf) or [.tex](./lab.tex)
    * [stb_image.h](./stb_image.h), [stb_image_write.h](./stb_image_write.h): libraries
    files for dealing with image files
    * [640x426.bmp](./640x426.bmp): sample image file (can also use your own)
    * [noRed.cpp](./noRed.cpp): kernel that removes red from the image;
    this file is included in the Colab notebook so you don't need this
    file unless you are running the lab outside the notebook

All material available for download from the [ToUCH git repository](https://github.com/TeachingUndergradsCHC/modules.git)  

### References 

1. <a name="pi4cs"></a>Peer Instruction for CS. <http://peerinstruction4cs.com/>
