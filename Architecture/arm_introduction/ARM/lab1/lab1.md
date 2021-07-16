## Lab: ARM vs. Thumb speed/codesize tradeoffs

### Description
In this lab you will complete a simple ARM program, compile it using
the GNU tool chain, and run the executable from the Linux command line.

### Environment

You will be building and running code from the Raspbian (Linux)
command line on your Raspberry Pi.

### Tools

Familiarize yourself with the following tools. 

  * `gcc`
  * `vim` or `emacs` or `nano`

### Instructions

#### 1. Download all the lab code 

Download the file lab1.s.  This contains some ARM assembly starter code
that you will complete and modify to compute the factorial function.
	
#### 2. Edit the file using your favorite editor

Open the file `lab1.s` using your editor.  This is an, incomplete, ARM
assembly program which will compute the factorial function for a given
input, `n`.  (That is, `n!`).  There are several things to note about the code.

##### a. Header information

You'll see several assembler "directives" at the beginning of the file.
These are telling the assembler that that C-library functions printf
and scanf will be used by this code, and that this code is implementing
the main() function.

##### b. First part of main()

The first part of the main() function is give to you.  This code prints
a prompt on the screen using printf(), and gets a single integer input
from the user using scanf().  This value, for `n` is stored in register
R0.

##### c. Loop code

Your job is to write a simple loop that will iterate from 1 to n and
compute `n!`.  You should follow the comments in the code as your guide.
Your instructor can assist you if you get stuck.  You should put your final
results, `n!` in register R2.

##### d. Epilouge code.

The last several instructions given to you in main() will print the value
of R2 to the terminal, and properly exit the function.

#### 3. Build and run your code using gcc

This lab utilizes the gnu toolchain to build your executable.  The code has
been set up so that you only need one command to assemble and link your program:

```
gcc lab1.s
```

This will produce an executable named `a.out`.  To run, simply type:

```
./a.out
```

A sample run is shown below:

```
> gcc lab1.s
> ./a.out
Enter n: 5
factorial is: 120
```


