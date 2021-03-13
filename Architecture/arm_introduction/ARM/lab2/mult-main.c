#include <stdio.h>
#include <time.h>
#define N 100

void multiply(int mat1[][N], int mat2[][N], int res[][N]);

int main() {
  int mat1[N][N]; 
  int mat2[N][N];
  int res[N][N]; // To store result
  int x;
  int i, j, k;
  clock_t CPU_time_1, CPU_time_2;
  for (k=0; k< 10; k++) {  // Run the entire algorithm 10 times
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
	mat1[i][j] = i;
	mat2[i][j] = i;
	res[i][j] = 0;
      }
    }
    CPU_time_1 = clock();
    for (x=0;x<100;x++)  // Perform 100 matrix multiplications
      multiply(mat1, mat2, res);
    CPU_time_2 = clock();
    printf("\nTime taken: %ld\n",CPU_time_2 - CPU_time_1);
  }
#if 0  // Uncomment to see results  
  printf("Result matrix is \n");
  for (i = 0; i < N; i++)
    {
      for (j = 0; j < N; j++)
	printf("%d ", res[i][j]);
      printf("\n");
    }
    printf("\nTime taken: %ld\n",CPU_time_2 - CPU_time_1);
#endif
  return 0;
}
