## Using Google Colab for CUDA programs

In order to run CUDA programs, students (and instructors) need access
to a system with an appropriate GPU and the development tools
installed.
The traditional approach to this is to install the hardware and
software to specific systems and allow remote access to those systems.
Unfortunately, this is not always easy-- someone needs to perform (and
maintain) these installations and students need to have access to
these systems (security policy at my institution makes such
access cumbersome).
Google Colab provides an alternative, with the ability to run programs
through a web interface based on Jupyter notebooks.
These notes aim to bring together information needed to use it (which
requires some installation steps).

The notes are an updated version of an 
[online post by Andrei Nechaev](https://medium.com/@iphoenix179/running-cuda-c-c-in-jupyter-or-how-to-run-nvcc-in-google-colab-663d33f53772).

### Instructions

1. Begin by going to Colab at
[https://colab.research.google.com](https://colab.research.google.com).

1. This creates a popup.
Create a new notebook using the option at the bottom.

1. Then tell the system that you want to use a GPU by selecting "Change
runtime type" in the Runtime menu and selecting GPU as the desired type of
hardware accelerator.

1. Copy the following code into the first cell of the notebook
    (without leading spaces) and hit the "play button":
    <pre>
    !git config --global url.\"https://github.com/\".insteadOf git://github.com/
    !pip install git+git://github.com/andreinechaev/nvcc4jupyter.git
    %load_ext nvcc_plugin
    </pre>
    This installs a plugin that lets you enter CUDA code in the notebook.

1. Create another cell by clicking "+ Code" directly below the menu.
    Then copy the following code into the new cell and hit the play
    button to run it:
    <pre>
    !sudo ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc
    !sudo ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++
    </pre>
    These make the system use version 5 of gcc and g++, which is the
    latest version that CUDA supports.

1. After this, you can enter the code to run in additional cells
    (created with "+ Code" and run with the play button) by
    preceding it with %%cu.
    For example, the following is a "Hello World" program:
    <pre>
    %%cu
    #include &lt;stdio.h>
    
    __global__ void hello() {
      int id = threadIdx.x + blockIdx.x * blockDim.x;
      printf("Hello from thread %d (%d of block %d)\n", id, threadIdx.x, blockIdx.x);
    }

    int main() {
      hello<<<5,4>>>();  //launch 5 blocks of 4 threads each
    
      cudaDeviceSynchronize();  //make sure kernel completes
    }
    </pre> 
