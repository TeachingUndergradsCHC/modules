/* Takes a list of consecutive numbers in an array
   and puts 0 in each position that is not prime
   using the Sieve of Eratosthenes algorithm. */

void sievemark(int *primes, int n)
{
  primes[1] = 0;
  int p = 2;
  while (p < n) {
    int index = 2;
    while(p * index < n) {
      primes[p * index] = 0;
      index++;
    }
    index = p + 1;
    while (index < n) {
      if (primes[index] != 0) {
	p = index;
	break;
      }
      index++;
    }
    if (index == n) return;
  }
  return;
}
