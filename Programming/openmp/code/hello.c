/* 
 * This program is used in the Hello World in OpenMP Demo 
 *  @author Apan Qasem 
 */

#include<stdio.h>
#include<omp.h>

void say_hello(unsigned n) {
  int i = 0;
  for (i = 0; i < n; i++)  
    printf("Hello World!\n");
  return;
}

void say_goodbye(unsigned n) {
  int i = 0;
  for (i = 0; i < n; i++)  
    printf("Goodbye World!\n");
  return;
}

int main() {

  omp_set_num_threads(4);

  #pragma omp parallel
  {

    //  uncomment following lines to demonstrate asynchronous execution of functions 
    //  say_hello(4);
    //  say_goodbye(4);

    int ID = omp_get_thread_num();
    printf("Hello World from %d!\n", ID);
    printf("Goodbye World from %d!\n", ID);
  }
  return 0;
}
