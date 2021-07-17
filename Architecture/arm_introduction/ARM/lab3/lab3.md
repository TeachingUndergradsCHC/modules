## Lab: Introduction to Neon

### Description
In this lab you will take a simple assembly program and rewrite it using
ARM Neon assembly instructions

### Environment

You will be building and running code from the Raspbian (Linux)
command line on your Raspberry Pi.

### Tools

Familiarize yourself with the following tools. 

  * `gcc`
  * `vim` or `emacs` or `nano`

### Instructions

You will be writing code to add the elements of two arrays together
storing the result in a third array.  Driver code is provided to initialize
the arrays with values, and a C implementation of the algorithm is
provided.  A linear assembly version of the array addition is also included.
You will write a SIMD version of the the assembly code using ARM Neon
assembly instructions.

#### 1. Download all the lab code 

You will be given three files:
  * main.c - sets up the three arrays, contains code to initialize and time the various implementations
  * add_arrays_linear.s - an assembly language implementation of the array addition code.
  * add_arrays_neon.s - a stub where you will write the neon version of the code.

#### 2. Write the NEON code.

In the editor of your choice, open up the file add_arrays_neon.s.  (You may
also want to consult the completed linear version of the code in
add_arrays_linear.s.)  The loop has been set up for you, and the addresses
of the three arrays are in registers R0, R1, and R2.  R3 contains the size
of the three arrays.  For this attempt, load array values into
the Neon 64-bit registers (d0, d1, etc.)  Then use an addition
instruction that treats the 64-bit registers as (eight) separate 8-bit values,
adding the 8-bit values together independently.  You will need to use Neon load, add, and store instructions.
Consult the lecture slides for details on the instructions.

#### 3. Build the code.

You will build the code directly from the command line using gcc.

`gcc -mfpu=neon -o neon1.out main.c add_arrays_linear.s add_arrays_neon.s`

The `-mfpu=neon` tells the compiler (assembler in this case) to accept Neon instructions and registers.
To run the code simply type: `./neon1.out`  The code will run all three
versions of the array addition code, C linear, assembly linear, and assembly
Neon.  Timings (in milliseconds) will be displayed for each of the three
functions.  After each timing, several elements from the result array will
be displayed a sort of check

#### 4. Improve 

In step three your NEON code added eight vector elements per instruction using
the 64-bit NEON registers.  NEON also allows you to pair two 64-bit registers
effectively creating a 128-bit registers.  SIMD instructions can be performed
on these register pairs, so in our code example we can perform 16 8-bit addition
operations using a single instruction.

Take your code from step three and modify it to use the 128-bit register pairs.
(We suggest you copy your add_arrays_neon.s to a backup file.)
Then build and run the code as you did above.

#### 5. Analysis

* You compiled and ran four different versions of vector addition, linear C, linear ASM, 64-bit NEON asm, and 128-bit NEON asm.  What improvements in performance did you see between these 4 versions?
* Why did the linear assembly code run so much faster than the compiled linear C code?
* Why did you not observe a factor of 8 speedup moving from the linear assembly code to the 64-bit NEON code?
* Why was the 128-bit NEON code not twice as fast at the 64-bit NEON code?
