TARGETS=quicksort_seq quicksort_par quicksort_hybrid
CC=gcc

all: $(TARGETS) 


quicksort_seq: quicksort_seq.c
	$(CC) -o $@ -O3 $<
quicksort_par: quicksort_par.c
	$(CC) -o $@ -fopenmp -O3 $<
quicksort_hybrid: quicksort_hybrid.c
	$(CC) -o $@ -fopenmp -O3 $< -fno-stack-protector -foffload=nvptx-none

clean:
	rm -rf $(TARGETS) 
