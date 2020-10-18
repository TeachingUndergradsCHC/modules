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

The notes are based on an
[online post by Andrei Nechaev](https://medium.com/@iphoenix179/running-cuda-c-c-in-jupyter-or-how-to-run-nvcc-in-google-colab-663d33f53772).

### Instructions

1. Begin by going to Colab at
[https://colab.research.google.com](https://colab.research.google.com).

1. This creates a popup.
Create a new notebook using the option at the bottom.

1. Then tell the system that you want to use a GPU by selecting "Change
runtime type" in the Runtime menu and selecting GPU as the desired type of
hardware accelerator.

1. Copy the following code into the first cell of the notebook and hit
the "play button" next:
    <pre>
    !apt update -qq;
    !wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb;
    !dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb;
    !apt-key add /var/cuda-repo-8-0-local-ga2/7fa2af80.pub;
    !apt-get update -qq;
    !apt-get install cuda gcc-5 g++-5 -y -qq;
    !apt install cuda-8.0;
    </pre>
    This step take several minutes since it involves installing the CUDA
    development tools into the notebook.
    It is the main drawback of using a Colab rather than a server, on
    which access would be quick once the tools are installed once.

1. Create another cell by clicking "+ Code" directly below the menu.
Then copy the following code into the new cell and hit the play button
to run it:

    `!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git`<br>
    `%load_ext nvcc_plugin`

    This installs a plugin that lets you enter CUDA code in the notebook.

1. Finally, run the following in another cell (created with "+ Code"
and run with the play button):

`!sudo ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc`<br>
`!sudo ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++`

These make the system use version 5 of gcc and g++, which is the
latest version that CUDA supports.

1. After this, you can enter the code to run in additional cells by
preceding it with %%cu 

