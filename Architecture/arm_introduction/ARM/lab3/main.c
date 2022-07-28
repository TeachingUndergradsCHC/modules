// This is lab3 of the Introduction to ARM module.
// This code sets up three arrays of bytes (unsigned characters)
// and sums the first two arrays putting the result in the
// third array.  The first call is made to a C function that adds
// the arrays element by element.  Then, two calls are made to functions implemented
// in ARM assembly.  The first is given and is is a simple
// loop that adds each element of the array individually
// putting the sums into the third array.  The second
// is to be developed by the students to use Neon instructions
// to speed up this operation.  Timing code is provided
// to time each of these.

#include <stdio.h>
#include <time.h>

#define SIZE 16384 // Size of arrays to manipulate, should be a power of 2
                   // at least 128

// put some initial values in x, and y, and set z array to all 0
void initialize_arrays(unsigned char *x, unsigned char *y, unsigned char *z)
{
  int i;
  for (i = 0; i < SIZE; i++) {
    x[i] = y[i] = i % 127;
    z[i] = 0;
  }
  return;
}

// This is a C version of adding the arrays element by element.
void add_arrays_linear_c(unsigned char *x, unsigned char *y, unsigned char *z, int size)
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
  
  clock_t CPU_time_1 = clock();
  add_arrays_linear_c(a, b, c, SIZE);
  clock_t CPU_time_2 = clock();
  printf("\nTime taken linear C: %ld\n",CPU_time_2 - CPU_time_1);
  printf("%i %i %i %i %i %i %i %i %i %i\n",c[0],c[1],c[2],c[3],c[10],c[11],c[63],c[64],c[65],c[66]);

  clock_t CPU_time_3 = clock();
  add_arrays_linear(a, b, c, SIZE);
  clock_t CPU_time_4 = clock();
  printf("\nTime taken linear asm: %ld\n",CPU_time_4 - CPU_time_3);
  printf("%i %i %i %i %i %i %i %i %i %i\n",c[0],c[1],c[2],c[3],c[10],c[11],c[63],c[64],c[65],c[66]);

  initialize_arrays(a, b, c);
  clock_t CPU_time_5 = clock();
  add_arrays_neon(a, b, c, SIZE);
  clock_t CPU_time_6 = clock();
  printf("\nTime taken Neon ASM: %ld\n",CPU_time_6 - CPU_time_5);
  printf("%i %i %i %i %i %i %i %i %i %i\n",c[0],c[1],c[2],c[3],c[10],c[11],c[63],c[64],c[65],c[66]);

}
