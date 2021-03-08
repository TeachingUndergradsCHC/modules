### [B1] Hybrid Algorithms   
Apan Qasem [\<apan@txstate.edu\>](mailto:apan@txstate.edu)

#### Exercises 

1. **[ Complexity Analysis ]** Consider the following hybrid parallel pseudocode for quick
   sort. Perform an asymptotic analysis of its runtime. 
   ```C++
    quicksort(array, start, end)
      if (start < end)
        middle = (start + end)/2;
        pivot = partition(array, start, end, middle)
	    target quickSort(values, start, pivot - 1)
        spawn quickSort(values, pivot + 1, end)
   ```

2. **[ Complexity Analysis ]** Two parallel algorithms for matrix-vector multiplication are shown
   below. The first uses conventional threads for parallelization while the second exploits achieves
   parallelization via target offloading. Analyze the complexity of the two algorithms. Show the derivation. 

   
   ```C++
     matvec(M, v, n, result)
       parallel for i = 1 to n
         for new j = 1 to n 
           result[i] = result[i] + M[i][j] * v[y]
   ```

   ```C++
     matvec(M, v, n, result)
       target for i = 1 to n
         for new j = 1 to n 
           result[i] = result[i] + M[i][j] * v[y]
   ```

  
3. **[ Experimental Analysis ]** Download the two parallel implementations of the matrix-vector
   multiplication from the repository. Run a performance experiment to empirically validate their 
   algorithmic complexity (see Problem 2). You can run the experiments on hopper.cs.X.edu which
   is set-up with the compiler and other performance tools. Run each implementation with increasing
   size and increasing number of threads. Plot their performance on two charts and discuss the
   results. 
  
   
