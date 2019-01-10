#include "./common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#define N 3
#define S 5
#define BR() printf("\n")

typedef struct _ARRAY{
  int x[S];
  int y[S];
}ARRAY;

__global__ void global_print(ARRAY* d_arrays){
  int i, j;
  for(j = 0 ; j < N ; j ++){
    printf("d_arrays[%d]:\n", j );
    for(i = 0; i < S; i++){
      d_arrays[j].x[i] = i * -1;
      d_arrays[j].y[i] = i * -1;
      printf("[%d] x: %d y : %d\n",i, d_arrays[j].x[i], d_arrays[j].y[i]);
    }//for
    BR();
  }//for
  BR();

}//func

void exec(){
  int i, j;
  ARRAY arrays[N];
  ARRAY *d_arrays;
  
  for(j = 0 ; j < N ; j ++){
    printf("arrays[%d]:\n", j );
    for(i = 0; i < S; i++){
      arrays[j].x[i] = i;
      arrays[j].y[i] = i + 1;
      printf("[%d] x: %d y : %d\n",i, arrays[j].x[i], arrays[j].y[i]);
    }//for
    BR();
  }//for
  BR();

  cudaMalloc(&d_arrays, sizeof(ARRAY) * N );
  cudaMemcpy(d_arrays, arrays, sizeof(ARRAY) * N, cudaMemcpyHostToDevice);
  global_print<<<1,1>>>(d_arrays);
  cudaMemcpy(arrays, d_arrays, sizeof(ARRAY) * N, cudaMemcpyDeviceToHost);
  cudaFree(d_arrays);
  for(j = 0 ; j < N ; j ++){
    printf("arrays[%d]:\n", j );
    for(i = 0; i < S; i++){
      printf("[%d] x: %d y : %d\n",i, arrays[j].x[i], arrays[j].y[i]);
    }//for
    BR();
  }//for
  BR();
}//func

int main(int argc, char const *argv[]) {
  exec();
  return 0;
}
