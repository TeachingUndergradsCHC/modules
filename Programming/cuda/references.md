## Introductory references about CUDA

Here are some resources for learning about CUDA.
They are suitable for giving to students as something to read/watch
before class or as a reference they can refer to in order to review
CUDA concepts after class.

  * [An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/),
    an NVIDIA blog post by Mark Harris from October 2012.
    Talks about the CUDA programming model and then talks through an
    example CUDA program that multiples a vector by a constant and
    adds it elementwise to a second vector.
  * [An Even Easier Intorduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/),
    an NVIDIA blog post by Mark Harris from January 2017.
    Despite the title, I don't think this is a simpler introduction
    than the one above.
    It assumes a bit more basic familiarity with CUDA, not introducing
    terminology like host and device.
    It uses vector addition as the basic example.
    Part of the example is a discussion of using multiple blocks,
    which is omitted from the "Easy Introduction".
    Also added is a demonstration of using nvprof to profile the
    program and understand its performance.

If you discover other useful resources that might be suitable for
inclusion in this list, I'm interested in hearing
about them.
