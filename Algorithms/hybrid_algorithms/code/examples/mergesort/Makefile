TARGETS=mergesort_seq mergesort_par mergesort_hybrid
CC=g++-10

all: $(TARGETS) 


mergesort_seq: mergesort_seq.c
	$(CC) -o $@  $<
mergesort_par: mergesort_par.c
	$(CC) -o $@ -fopenmp  $<
mergesort_hybrid: mergesort_hybrid.c
	$(CC) -o $@ -fopenmp  $< -fcf-protection=none  -fno-stack-protector -foffload=nvptx-none

clean:
	rm -rf $(TARGETS) 
