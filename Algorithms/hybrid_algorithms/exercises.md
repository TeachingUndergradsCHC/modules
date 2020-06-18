### [B1] Hybrid Algorithms 
Apan Qasem [\<apan@txstate.edu\>](mailto:apan@txstate.edu)

#### Exercises 

1. **[ Complexity Analysis ]** Consider the following hybrid parallel pseudocode for quick
   sort. Perform an asymptotic analysis of its runtime. 


    ```
	   quicksort(array, start, end)  
         if (start < end) 
           middle = (start + end)/2;
           pivot = partition(array, start, end, middle)
          
		 task quickSort(values, start, pivot - 1)
         spawn quickSort(values, pivot + 1, end)
	     sync
    ```
    



