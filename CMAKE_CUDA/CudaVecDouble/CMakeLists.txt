find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("-- CUDA --")
    message("Version: " ${CUDA_VERSION})
    message("Library: " ${CUDA_CUDA_LIBRARY})
    message("Runtime: " ${CUDA_CUDART_LIBRARY})
    CUDA_ADD_LIBRARY(CudaVecDouble STATIC
        CudaVecDouble.cu
        CudaVecDouble.h
    )
    set(CUDA_NVCC_FLAGS "-O2 -arch=sm_61 -lcudadevrt --relocatable-device-code true --ptxas-options=-v")
endif()
