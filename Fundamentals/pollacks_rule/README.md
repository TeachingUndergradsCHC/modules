## Pollack's Rule as a Justification for Heterogeneous Computing
David Bunde [\<dbunde@knox.edu\>](mailto:dbunde@knox.edu)

### Description

This module introduces Pollack's rule, that the performance of a core
is proportional to the square root of its area.  This rule implies
that the peak performance of a chip is maximized by cutting it into
cores that are as small as possible.  Pushing against this is Amdahl's
law, which means that higher peak performance will not improve
performance in the presence of serial sections in the code.  This
leads to the conclusion that having a variety of core sizes is
preferable (i.e.a heterogeneous system), with larger cores available
for serial sections and many small cores available for sections with
abundant parallalism. 

### Context

The module is intended to motivate heterogeneous computing for
students who have already seen the argument for parallel computing.
The module would fit into a typical Computer Organization course since it has
a hardware focus.  I use it in an Introduction to Systems course.
Because it has a very high-level point of view and doesn't rely on
details of the hardware, it could also go into an early (e.g. CS 1 or 2)
course that uses some of the other Fundamentals modules.

I teach this module using the Peer Instruction pedagogy, in which
lectures feature multiple choice questions that the students answer
and discuss.
(See [[1]](#pi4cs) for more information.)
Because not everyone teaches this way, the module includes 2 sets of
slides, one for Peer Instruction and one for a more conventional
lecture class.

### Topics

Heterogeneous Computing topics covered in this module are listed
below.  Bloom's classification is shown in brackets.

  * Amdahl's Law [A]
  * Pollack's Rule [A]

### Learning Outcomes

Having completed this module, students should be able to

  * Explain the performance advantage of a processor having heterogeneous core sizes
  * Use Pollack's rule and Amdahl's Law to estimate the performance of
  a program when the program's amount of parallelizability and the
  processor's configuration of cores are specified

### Instructor Resources

This module includes the following teaching materials:

  * Slides if using Peer Instruction
    ([.pptx](./lecture_slides_pi.pptx), [.pdf](./lecture_slides_pi.pdf))
  * Slides for a traditional lecture
    ([.pptx](./lecture_slides.pptx), [.pdf](./lecture_slides.pdf))
  * [Pedagogical notes](pedagogy_notes.md): Suggestions based on
    previous experience teaching the module
  * [Reference material](./reference_material.md): Resources for further reading

All materials available for download from the
[ToUCH git repository](https://github.com/TeachingUndergradsCHC/modules.git)

### References

1. <a name="pi4cs"></a>Peer Instruction for CS. <http://peerinstruction4cs.com/>
