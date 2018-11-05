CU_APPS= getDeviceProperties\

C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_61 -o $@ $< -lcudadevrt --relocatable-device-code true --ptxas-options=-v
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
