#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void sievemark(int *primes, int n);

int main(int argc, char **argv)
{
  int size = 50000;  // default size of array
  int *primes;       // Array of primes to check
  int i, j;
  if (argc > 1)    // Size of array can be passed as first param on command line
    size = atoi(argv[1]);
  if (size < 2) {
    fprintf(stderr,"Size must be greater than 1\n");
    return -1;
  }
  primes = (int *)malloc(size * sizeof(int));
  for(i = 0; i < size; i++)
    primes[i] = i;
  for(j=0;j<10;j++) {   // run code 10 times
    clock_t CPU_time_1 = clock();
    sievemark(primes, size);
    clock_t CPU_time_2 = clock();
    printf("\nTime taken: %ld\n",CPU_time_2 - CPU_time_1);
  }
#if 0  // uncomment to see result printed
  for(i = 2; i < size; i++)
    if (primes[i]) printf("%d ",i);
#endif
  return 0;
}
