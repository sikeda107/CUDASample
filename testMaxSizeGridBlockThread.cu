#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ int d_N;
__device__ short int d_array[5][5];

__global__ void Kernel_add_one(){
  int thidx = threadIdx.x;
  int thidy = threadIdx.y;
  int blid = blockIdx.x;

  if(thidx < d_N && thidy < d_N && blid < 1){
    printf("x%d y%d\n",thidx,thidy );
      d_array[thidy][thidx]++;
  }
  else{}
}//end-kernel

__global__ void Kernel_print_array(){
  printf("d_N: %d\n",d_N );
  printf("%s\n","device array" );
  for (size_t i = 0; i < d_N; i++) {
    for (size_t j = 0; j < d_N; j++) {
        printf("%d ",d_array[i][j]);
    }
    printf("\n");
  }
}//end-kernel

int main(void)
{
    int dev_num =0;
    int N = 5;
    short int array[5][5];
    dim3 grid,block;

    //----set up device START-----
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev_num);
    printf("Using Device %d:%s\n",dev_num,deviceProp.name);
    cudaSetDevice(dev_num);
    //----set up device END-----

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
          array[i][j] = i * N + j;
      }
    }

    printf("%s\n","before array" );
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
          printf("%d ",array[i][j] );
      }
      printf("\n");
    }

    // copy the host variable to the global
    CHECK(cudaMemcpyToSymbol(d_N, &N, sizeof(int)));
    // CHECK(cudaMemcpyToSymbol(d_array, &array[0][0][0], sizeof(short int) * (N * N * N)));
    CHECK(cudaMemcpyToSymbol(d_array, array, sizeof(short int) * (N * N)));

    grid.x = 1; grid.y = 1;
    block.x = 1; block.y = 1;

    // invoke the kernel
    Kernel_print_array<<<grid, block >>>();
    cudaDeviceSynchronize();

    grid.x = 1; grid.y = 1;
    // block.x = 204; block.y = 5; // number of threads = 1020 (< 1024) OK
    block.x = 73; block.y = 25; // number of threads = 1025 (> 1025) NG

    Kernel_add_one<<< grid, block >>>();
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    grid.x = 1; grid.y = 1;
    block.x = 1; block.y = 1;
    Kernel_print_array<<< grid, block >>>();

    // copy the global variable back to the host
    // CHECK(cudaMemcpyFromSymbol(&array[0][0][0], d_array, sizeof(short int) * (N * N * N)));
    CHECK(cudaMemcpyFromSymbol(array, d_array, sizeof(short int) * ( N * N)));

    printf("%s\n","after array" );
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
          printf("%d ",array[i][j]);
      }
      printf("\n");
    }
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
