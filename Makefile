CU_APPS= getDeviceProperties testStaticGlobalMemory testMaxSizeGridBlockThread \
				sumMatrix1D sumMatrix2D_1 sumMatrix2D_2 cudaMemcpyOnDevice TestcdpSimpleQuickSort\

C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_61 -o $@ $< -lcudadevrt --relocatable-device-code true --ptxas-options=-v
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
