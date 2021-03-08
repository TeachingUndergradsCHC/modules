#include <iostream>

using namespace std;

int main(){
  const int N = 1000;
  int a[N];

  for(unsigned i = 0; i < N; i++)
    a[i] = 1;

  #pragma omp target teams distribute parallel for map(tofrom:d[0:N])
  for(unsigned i = 0;i < N; i++)
    a[i] *= 3 * i + 1;

  for(unsigned i = 0; i < 1; i++)
    cout << "Result a[0] = " << a[i] << endl;
  return 0;
}
