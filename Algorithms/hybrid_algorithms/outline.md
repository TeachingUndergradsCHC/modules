#### Outline 

1. Heterogeneous Computing 
    - current landscape and future of heterogeneous computing 
    - examples of different classes of heterogeneity 
2. Accelerator Offloading 
     - most common form of heterogeneity (and the focus of this module) 
	 - examples: CPU + GPU, CPU + FPGA, CPU + Quantum 
	 - future outlook: CPU + many different accelerators on the same system  		 
3. Hybrid Algorithms 
    - DAG with task offloading 
	- contrast with DAG for sequential and parallel algorithms 
4. Heterogeneous Parallel Programming Models 
    - dynamic multithreading review (CLRS 27)
    - **task** keyword to express heterogeneity 	
    - example: recursive Fibonacci 
5. Divide-and-Conquer 
    - general template + complexity analysis 
    - opportunities for parallelism
6. Merge Sort 
    - merge sort as an example of parallel divide-and-conquer
	- complexity analysis 
7. Merge Sort with Acceleration 
    - task offloading in merge sort 
	- complexity analysis 
8. Practical Considerations 
    - Identifying tasks for offloading 
    - Data Movement
	- Scheduling
9. Hands-on Example 
    - GPU acceleration with the **task** directive in OpenMP

