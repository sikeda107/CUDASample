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

#define N 32
#define M 32

__device__ short int d_array[N * M];
__device__ short int d_array_tmp[N];
__device__ int d_sum2;

__global__ void Kernel_sum_array1D(){

  // extern __shared__ int sdata[];
  __shared__ short int sdata[N];

  if(threadIdx.x < N && blockIdx.x < 1){
    sdata[threadIdx.x]= d_array_tmp[threadIdx.x];
    __syncthreads();
    printf("thidx %d sdata %d\n",threadIdx.x, sdata[threadIdx.x]);
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      if (threadIdx.x < s && (threadIdx.x + s) < N) {
        sdata[threadIdx.x] += sdata[threadIdx.x  + s];
      }
      __syncthreads();
      if (threadIdx.x == 0){
        for (size_t i = 0; i < N; i++) {
          printf("%d ",sdata[i]);
        }
        printf("\n" );
      }
    }//for

    if (threadIdx.x == 0){
      // for (size_t i = 0; i < N; i++) {
      //   printf("%d ",sdata[i]);
      // }
      d_sum2 = sdata[0];
    }//if

  }//if
  else{}
}//end-kernel

__global__ void Kernel_sum_array2D(){

  dim3 grid,block;

  // Assumption of N blocks M threads
  int index = blockIdx.x * M  + threadIdx.x ;
  // extern __shared__ int sdata[];
  __shared__ short int sdata[M];
  if(threadIdx.x < M && blockIdx.x < N){
    sdata[threadIdx.x]= d_array[index];
    __syncthreads();
    printf("blidx %d thidx %d sdata %d\n",blockIdx.x , threadIdx.x , sdata[threadIdx.x]);
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    // for (unsigned int s = (M+1)/ 2; s > 0 ; s>>=1) {
      if (threadIdx.x  < s && (threadIdx.x + s) < M) {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
      }
      __syncthreads();
    }//for

    if (threadIdx.x == 0){
      d_array_tmp[blockIdx.x] = sdata[0];
    }//if
  }//if
  else{}
}//end-kernel

__global__ void Kernel_print_array(){
  printf("\nN: %d, M:%d\n", N, M );
  printf("%s\n","device array" );
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      printf("%d ",d_array[i * M +j]);
    }
    printf("\n");
  }
}//end-kernel

int main(void)
{
    int dev_num =0;
    int sum = 0, sum2 = 0;
    short int array[N * M];
    short int array_tmp[N];
    double iStart, iElaps;
    dim3 grid,block;

    // start timer
    iStart = seconds();

    //----set up device START-----
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev_num);
    printf("Using Device %d:%s\n",dev_num,deviceProp.name);
    cudaSetDevice(dev_num);
    //----set up device END-----

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {
        array[i * M + j] = i + 1;
        sum += array[i * M + j];
      }
    }

    printf("\n%s\n","before array" );
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {
        printf("%d ",array[i * M +j]);
      }
      printf("\n");
    }

    // copy the host variable to the global
    CHECK(cudaMemcpyToSymbol(d_array, array, sizeof(short int) * N * M ));

    grid.x = 1; grid.y = 1;
    block.x = 1; block.y = 1;

    // invoke the kernel
    Kernel_print_array<<< grid, block >>>();
    cudaDeviceSynchronize();

    grid.x = 32; grid.y = 1;
    block.x = 32; block.y = 1; // if number of threads is (< 1024), OK

    // Kernel_sum_array<<< grid, block, sizeof(int) * N * M >>>();
    Kernel_sum_array2D<<< grid, block >>>();
    cudaDeviceSynchronize();
    // Kernel_sum_array_nonshared<<< grid, block >>>();
    // cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    Kernel_sum_array1D<<< grid, block >>>();
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // copy the global variable back to the host
    // CHECK(cudaMemcpyFromSymbol(&array[0][0][0], d_array, sizeof(short int) * (N * N * N)));
    CHECK(cudaMemcpyFromSymbol(array, d_array, sizeof(short int) * N * M ));
    CHECK(cudaMemcpyFromSymbol(array_tmp, d_array_tmp, sizeof(short int) * N ));
    CHECK(cudaMemcpyFromSymbol(&sum2, d_sum2, sizeof(int) ));

    printf("\n%s\n","after array" );
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {
        printf("%d ",array[i * M +j]);
      }
      printf("\n");
    }
    printf("\n%s\n","after array_tmp" );
    for (size_t i = 0; i < N; i++) {
      printf("%d ",array_tmp[i]);
      // sum2+=array_tmp[i];
    }
    printf("\n");

    printf("\nsum: %d, sum2: %d \n",sum, sum2);
    CHECK(cudaDeviceReset());

    iElaps = seconds() - iStart;
    printf("line: %d iElaps: %lf\n",__LINE__, iElaps );
    return EXIT_SUCCESS;
}
