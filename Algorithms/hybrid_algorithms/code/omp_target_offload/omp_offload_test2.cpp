#include <iostream>

using namespace std;

bool search(int *a, int n, int val) {
  for (unsigned i = 0; i < n; i++)
    if (a[i] == val)
      return true;
  return false;
}

int main(){
  const int N=1000;
  int d[N];

  for(auto i=0;i<N;i++)
    d[i] = 1;

  bool result;
  #pragma omp target
  result = search(d, N, 17);

  if (result) 
    cout << "Found!" << endl;
  else
    cout << "Search Failed!" << endl;
    
  return 0;
}
