## [C1] Heterogeneous Architectures I
Phil Schielke [\<Philip.Schielke@concordia.edu\>](philip.schielke@concordia.edu)

### Description 
This module introduces students to the basics of the ARM architecture.
Many students are introduced to MIPS or some other real-world architecure during
the course of their studies but are never exposed to another CPU.  The
goal in this module is
to introduce students do a different architecture so they can think
through how different architectures may function.  The benefit of introducing
ARM is that many ARM-based devices have several forms of heterogeneity
within a single  part, including a vector floating point unit, the NEON integer SIMD
system, and Thumb mode which forces the programmer to
consider speed/codesize/power tradeoffs.

### Context 

This module is intended for a first course in an undergraduate computer organization course, which is typically 
taken during freshman or sophomore year.  As such, it is expected that students have the following background
when taking such a class before the module is presented.
  * basic programming experience in a language like C or C++
  * basic understanding of at least one hardware ISA, and simple programming at the assembly level.

### Topics 

HC topics covered in this module are listed below. Bloom's classification is shown in brackets

* Alternate ISAs [A]
* Codesize/Speed tradeoffs [C]
* Single-chip Heterogeneity [K]
* SIMD Hardware [K]

### Learning Outcomes

Having completed this module, students should be able to: 

* Describe some of the differences between the MIPS and ARM ISAs
* Explain the basics of SIMD (NEON and VFP)
* Measure the codesize/speed tradeoff in ARM programs

### Instructor Resources 

  * [Slide Deck](./ARM/ARM_intro_lecture.pptx): Brief introduction to ISA
  noting differences with MIPS
  * [Slide Deck](./ARM/Thumb_intro.pptx): Introduction to ARM's Thumb mode
  * [Slide Deck](./ARM/NEON_intro.pptx): Introduction to ARM's NEON SIMD coprocessor
  * [Instructor Notes](./ARM/ARM_intro_lecture_notes.docx):  Instructor
    notes for above slide decks.
  * [Lab](./ARM/lab1/lab1.md): A lab designed to introduce students to the ARM ISA
  * [Lab](./ARM/lab2/lab2.md): A lab where students can experiment with the tradeoffs between ARM, Thumb1, and Thumb2 code
  * [Lab](./ARM/lab3/lab3.md): A lab in which students rewrite some assembly code to utilize ARM NEON instructions.
  * [ARM Cheatsheet](./ARM/ARM_cheatsheet.pdf):  ARM ISA cheatsheet
  * [MIPS Cheatsheet](./MIPS/MIPS_cheatsheet.pdf): MIPS ISA cheatsheet	
  * [Reference Material](./reference_material.md): additional resources for instructors




