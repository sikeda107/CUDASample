// TestcdpSimpleQuickSort.c

#include <stdio.h>
//CUDA library
#include<cuda_runtime.h>
#include "./common/common.h"
#include "./cdpSimpleQuickSort.h"

void test_qsort(){
  int array[5][50];
  int *d_array;

  //----set up device START-----
  int dev_num =0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,dev_num);
  printf("Using Device %d:%s\n",dev_num,deviceProp.name);
  cudaSetDevice(dev_num);
  //----set up device END-----
  srand(10);

  printf("Before\n");
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 50; j++) {
      array[i][j] = rand() % 50;
      printf("%d ",array[i][j] );
    }
    printf("\n");
  }
  printf("\n");

  cudaMalloc((int**)&d_array, sizeof(array));
  cudaMemcpy(d_array, array, sizeof(array), cudaMemcpyHostToDevice);

  for (size_t i = 0; i < 5; i++) {
    run_qsort(&d_array[i * 50], 50);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(array, d_array, sizeof(array), cudaMemcpyDeviceToHost);
  printf("After\n");
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 50; j++) {
      printf("%d ",array[i][j] );
    }
    printf("\n");
  }
  printf("\n");
  cudaFree(d_array);
}//function

int main(int argc, char *argv[]){

  test_qsort();
  return 0;
}
