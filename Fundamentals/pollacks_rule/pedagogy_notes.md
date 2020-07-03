## Pedagogical Notes for Pollack's Rule Module
David Bunde [\<dbunde@knox.edu\>](mailto:dbunde@knox.edu)

## Relating to Specific Systems

Since the module is somewhat abstract, it's important to relate the
material to something concrete.  A nice example of this is cell
phones, many of which use heterogeneous cores.  (Note that this is not
universal-- I have an older phone, which has only a single type of core.)
That is the purpose of the last slide, which presents the
configuration of a student's phone.
This was found for the specific phone using a benchmarking app.
Because that was an Android phone, we used Geekbench 5 (available in
the Google Play store).  This costs money on an iPhone, but Antutu has
a free alternative.

Note that this is a slightly different kind of heterogeneity than
discussed in the module; the performance of cores on the phone is
reported in terms of their clock frequency rather than their size.
Achievable clock frequency is a function of power so it is still an
example of splitting a limited resource in a non-uniform way.

## Assignments

To check student understanding of this material, I typically
assign some homework problems where the students calculate the running
time under different configurations.  (For example: Consider a
processor with one core of size 1/2 and three cores of size 1/6.
Assuming Pollack's rule relating the size of a core and its
performance holds, calculate (a) the peak performance of this
processor, and (b) the expected performance of a program that can be
80% parallelized.)

Another variation of the computation that you could look at (in
lecture or as homework) is parallel parts have a bounded degree of
parallelism, meaning they can only use that number of cores.
(For example, the parallel part only creates 2 threads so it cannot
utilize more than 2 cores.)
Then the performance on that part would be the performance using that
number of cores (presumably the fastest ones) rather than using all
the cores.
Parallel sections with bounded parallelism like this could replace
serial sections or be present in addition to them.
