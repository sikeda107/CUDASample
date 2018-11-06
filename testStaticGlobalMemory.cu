#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ int d_N;
__device__ short int d_array[3][3][3];

__global__ void Kernel_add_one(){
  int thid = threadIdx.x;
  int blid = blockIdx.x;

  if(thid < d_N && blid < d_N){
    for (size_t k = 0; k < d_N; k++) {
      d_array[blid][thid][k]++;
    }
  }
  else{}
}//end-kernel

__global__ void Kernel_print_array(){
  printf("d_N: %d\n",d_N );
  printf("%s\n","device array" );
  for (size_t i = 0; i < d_N; i++) {
    for (size_t j = 0; j < d_N; j++) {
      for (size_t k = 0; k < d_N; k++) {
        printf("%d ",d_array[i][j][k] );
      }
    }
    printf("\n");
  }
}//end-kernel

int main(void)
{
    int dev_num =0;
    int N = 3;
    short int array[3][3][3];

    //----set up device START-----
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev_num);
    printf("Using Device %d:%s\n",dev_num,deviceProp.name);
    cudaSetDevice(dev_num);
    //----set up device END-----

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
          array[i][j][k] = j;
        }
      }
    }

    printf("%s\n","before array" );
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
          printf("%d ",array[i][j][k] );
        }
      }
      printf("\n");
    }

    // copy the host variable to the global
    CHECK(cudaMemcpyToSymbol(d_N, &N, sizeof(int)));
    // CHECK(cudaMemcpyToSymbol(d_array, &array[0][0][0], sizeof(short int) * (N * N * N)));
    CHECK(cudaMemcpyToSymbol(d_array, array, sizeof(short int) * (N * N * N)));

    // invoke the kernel
    Kernel_print_array<<<1, 1 >>>();
    cudaDeviceSynchronize();
    Kernel_add_one<<<N, N >>>();
    cudaDeviceSynchronize();
    Kernel_print_array<<<1, 1 >>>();

    // copy the global variable back to the host
    // CHECK(cudaMemcpyFromSymbol(&array[0][0][0], d_array, sizeof(short int) * (N * N * N)));
    CHECK(cudaMemcpyFromSymbol(array, d_array, sizeof(short int) * (N * N * N)));

    printf("%s\n","after array" );
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        for (size_t k = 0; k < N; k++) {
          printf("%d ",array[i][j][k] );
        }
      }
      printf("\n");
    }
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
