TARGETS=omp_offload_test0 omp_offload_test1 omp_offload_test2
CXX=g++

OMP_OFFLOAD_FLAGS=-fno-stack-protector -foffload=nvptx-none
all: $(TARGETS) 

omp_offload_test0: omp_offload_test0.cpp
	$(CXX) -o $@ -O3 $< $(OMP_OFFLOAD_FLAGS)
omp_offload_test1: omp_offload_test1.cpp
	$(CXX) -o $@ -fopenmp -O3 $< $(OMP_OFFLOAD_FLAGS) 
omp_offload_test2: omp_offload_test2.cpp
	$(CXX) -o $@ -fopenmp -O3 $< $(OMP_OFFLOAD_FLAGS) 

clean:
	rm -rf $(TARGETS) 
