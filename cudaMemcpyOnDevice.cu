#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000*10

__global__ void Kernel_Memcpy(int* dst, int* src){
  int thid = threadIdx.x;
  int blid = blockIdx.x;
  int i;

  if(thid < 1 && blid < 1){

    printf("device src:\n");
    for ( i = 0; i < N; i++) {
      printf("%d ", src[i]);
    }
    printf("\n");

    printf("device dst:\n");
    for ( i = 0; i < N; i++) {
      printf("%d ", dst[i]);
    }
    printf("\n");
    printf("cudaMemcpyAsync()\n");
    cudaMemcpyAsync(dst, src, sizeof(int) * N ,cudaMemcpyDeviceToDevice);
    // for ( i = 0; i < N; i++) {
    //   dst[i] = src[i];
    // }
    printf("device src:\n");
    for ( i = 0; i < N; i++) {
      printf("%d ", src[i]);
    }
    printf("\n");

    printf("device dst:\n");
    for ( i = 0; i < N; i++) {
      printf("%d ", dst[i]);
    }
    printf("\n");
  }
  else{}
}//end-kernel

__global__ void Kernel_print_array(int *array){
  int i;
  printf("N: %d\n",N );
  printf("%s\n","device array" );
  for ( i = 0; i < N; i++) {
    printf("%d ", array[i]);
  }
  printf("\n");
}//end-kernel

int main(void){

  int *h_src, *h_dst;
  int *d_src, *d_dst;
  //----set up device START-----
  int dev_num =0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,dev_num);
  printf("Using Device %d:%s\n",dev_num,deviceProp.name);
  cudaSetDevice(dev_num);
  //----set up device END-----

  h_src = (int*)malloc(sizeof(int) * N);
  h_dst = (int*)malloc(sizeof(int) * N);

  memset(h_src, 0, sizeof(int) * N);
  memset(h_dst, 0, sizeof(int) * N);

  printf("host src:\n");
  for (size_t i = 0; i < N; i++) {
    h_src[i] = i+1;
    printf("%d ", h_src[i]);
  }
  printf("\n");

  printf("host dst:\n");
  for (size_t i = 0; i < N; i++) {
    printf("%d ", h_dst[i]);
  }
  printf("\n");

  cudaMalloc((int**)&d_src, sizeof(int)*N);
  cudaMalloc((int**)&d_dst, sizeof(int)*N);
  cudaMemcpy(d_src, h_src, sizeof(int)*N, cudaMemcpyHostToDevice);
  printf("Kernel_Memcpy<<<>>>()\n");
  Kernel_Memcpy<<<1,1>>>(d_dst, d_src);
  cudaMemcpy(h_dst, d_dst, sizeof(int)*N, cudaMemcpyDeviceToHost);

  printf("host src:\n");
  for (size_t i = 0; i < N; i++) {
    printf("%d ", h_src[i]);
  }
  printf("\n");

  printf("host dst:\n");
  for (size_t i = 0; i < N; i++) {
    printf("%d ", h_dst[i]);
  }
  printf("\n");
  CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
