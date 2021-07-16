CC = gcc
CFLAGS = -fopenmp 
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_35 --Wno-deprecated-gpu-targets

TARGETS = big_wkld little_wkld cache_wkld gpu_wkld

all: $(TARGETS)

%: %.c
	$(CC) -o $@ $(CFLAGS) $<

%: %.cu
	$(NVCC) -o $@ $(CUDA_FLAGS) $<

clean:
	rm -rf *.o $(TARGETS) 
install:
	rm -rf wk?
	ln -s big_wkld wk0
	ln -s little_wkld wk1
	ln -s gpu_wkld wk2
	ln -s cache_wkld wk3
