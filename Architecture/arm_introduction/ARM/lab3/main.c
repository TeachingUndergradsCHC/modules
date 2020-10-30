// This is lab3 of the Introduction to ARM module.
// This code sets up three arrays of bytes (unigned characters)
// and sums the first two arrays putting the result in the
// third array.  Two calls are made to functions implemented
// in ARM assembly.  The first is given and is is a simple
// loop that adds each element of the array individually
// putting the sums into the third array.  The second
// is to be developed by the students to use Neon instructions
// to speed up this operation.  Timeing code is provided
// to time each of these.
#include <stdio.h>

#define SIZE 1024 // Size of arrays to manipulate

void initialize_arrays(unsigned char *x, unsigned char *y, unsigned char *z)
{
  int i;
  for (i = 0; i < SIZE; i++) {
    x[i] = y[i] = i % 127;
    z[i] = 0;
  }
  return;
}

void add_arrays_linear(unsigned char *x, unsigned char *y, unsigned char *z, int size)
{
  int i;
  for (i = 0; i < size; i++)
    z[i] = x[i] + y[i];
  return;
}

void add_arrays_neon(unsigned char *x, unsigned char *y, unsigned char *z, int size)
{
  int i;
  for (i = 0; i < size; i++)
    z[i] = x[i] + y[i];
  return;
}

int main()
{
  unsigned char a[SIZE],b[SIZE],c[SIZE];
  
  initialize_arrays(a, b, c);
  // timing code start
  add_arrays_linear(a, b, c, SIZE);
  // time code end
  printf("%i %i %i %i %i %i %i %i %i %i\n",c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]);

  initialize_arrays(a, b, c);
  // timing code start
  add_arrays_neon(a, b, c, SIZE);
  // timing code end
  printf("%i %i %i %i %i %i %i %i %i %i\n",c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9]);
}
