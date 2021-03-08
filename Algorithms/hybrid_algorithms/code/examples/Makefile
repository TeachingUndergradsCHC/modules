TARGETS=quicksort_seq quicksort_par quicksort_hybrid
CXX=g++

all: $(TARGETS) 


quicksort_seq: quicksort_seq.cpp
	$(CXX) -o $@ -O3 $<
quicksort_par: quicksort_par.cpp
	$(CXX) -o $@ -fopenmp -O3 $<
quicksort_hybrid: quicksort_hybrid.cpp
	$(CXX) -o $@ -fopenmp -O3 $< -fno-stack-protector -foffload=nvptx-none

clean:
	rm -rf $(TARGETS) 
