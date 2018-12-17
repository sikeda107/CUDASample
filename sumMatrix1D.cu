#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// reference :
// http://gpu-computing.gsic.titech.ac.jp/Japanese/Lecture/2010-06-28/reduction.pdf
// GPUコンピューティング
// （CUDA）講習会
// CUDAプログラムの最適化
// 東工大学術情報センター　丸山直也
// 2010/06/28 

__device__ int d_N;
__device__ int d_sum;
__device__ short int d_array[21];

__global__ void Kernel_sum_array(){
  int thidx = threadIdx.x;
  int blid = blockIdx.x;
  extern __shared__ int sdata[];

  if(thidx < d_N && blid < 1){
    sdata[thidx]= d_array[thidx];
    __syncthreads();
    printf("thidx %d sdata %d\n",thidx, sdata[thidx]);
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (thidx < s) {
        sdata[thidx] += sdata[thidx + s];
      }
      __syncthreads();
    }//for

    if (thidx == 0){
      d_sum = sdata[0];
    }//if

  }//if
  else{}
}//end-kernel

__global__ void Kernel_sum_array_nonshared(){
  int thidx = threadIdx.x;
  int blid = blockIdx.x;

  if(thidx < d_N && blid < 1){
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (thidx < s) {
        d_array[thidx] += d_array[thidx + s];
      }
      __syncthreads();
    }//for

    if (thidx == 0){
      d_sum = d_array[0];
    }//if

  }//if
  else{}
}//end-kernel

__global__ void Kernel_print_array(){
  printf("\nd_N: %d\n",d_N );
  printf("%s\n","device array" );
  for (size_t i = 0; i < d_N; i++) {
    printf("%d ",d_array[i]);
  }
}//end-kernel

int main(void)
{
    int dev_num =0;
    int N = 21, sum = 0, sum2 = 0;
    short int array[21];
    dim3 grid,block;

    //----set up device START-----
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev_num);
    printf("Using Device %d:%s\n",dev_num,deviceProp.name);
    cudaSetDevice(dev_num);
    //----set up device END-----

    for (size_t i = 0; i < N; i++) {
          array[i] = i + 1;
          sum += array[i];
    }

    printf("\n%s\n","before array" );
    for (size_t i = 0; i < N; i++) {
      printf("%d ",array[i]);
    }

    // copy the host variable to the global
    CHECK(cudaMemcpyToSymbol(d_N, &N, sizeof(int)));
    // CHECK(cudaMemcpyToSymbol(d_array, &array[0][0][0], sizeof(short int) * (N * N * N)));
    CHECK(cudaMemcpyToSymbol(d_array, array, sizeof(short int) * N ));

    grid.x = 1; grid.y = 1;
    block.x = 1; block.y = 1;

    // invoke the kernel
    Kernel_print_array<<< grid, block >>>();
    cudaDeviceSynchronize();

    grid.x = 32; grid.y = 1;
    block.x = 32; block.y = 1; // if number of threads is (< 1024), OK

    Kernel_sum_array<<< grid, block, sizeof(int) * N >>>();
    cudaDeviceSynchronize();
    // Kernel_sum_array_nonshared<<< grid, block >>>();
    // cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    grid.x = 1; grid.y = 1;
    block.x = 1; block.y = 1;
    Kernel_print_array<<< grid, block >>>();

    // copy the global variable back to the host
    // CHECK(cudaMemcpyFromSymbol(&array[0][0][0], d_array, sizeof(short int) * (N * N * N)));
    CHECK(cudaMemcpyFromSymbol(array, d_array, sizeof(short int) * N));
    CHECK(cudaMemcpyFromSymbol(&sum2, d_sum, sizeof(int)));

    printf("\n%s\n","after array" );
    for (size_t i = 0; i < N; i++) {
      printf("%d ",array[i]);
    }
    printf("\nsum: %d, sum2: %d \n",sum, sum2);
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
