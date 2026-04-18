max-cut: max-cut.c max-cut.cu
	mpixlc -O3 max-cut.c -c -o max-cut-mpi.o
	nvcc -O3 -arch=sm_70 max-cut.cu -c -o max-cut-cuda.o 
	mpixlc -O3 max-cut-mpi.o max-cut-cuda.o -o max-cut -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ 
