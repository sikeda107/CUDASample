#include <stdio.h>
#include <stdlib.h>

void exec();

int main(int argc, char const *argv[]) {
  exec();
  return 0;
}
__global__ void kernel_checkvalue(){
  int x = 0;
  // __shared__ int x;
  x  = 1;
  if(threadIdx.x == 0){
    x++;
  }
  __syncthreads();
  printf("id: %d x: %d addr: %p %X\n", threadIdx.x, x, &x, &x);
  if(threadIdx.x == 1){
    x++;
  }
  __syncthreads();
  printf("id: %d x: %d addr: %p %X\n", threadIdx.x, x, &x, &x);
}
void exec(){
  int iDev = 0;
  dim3 grid, block;
  cudaDeviceProp iProp;
  cudaGetDeviceProperties(&iProp, iDev);
  printf("Device %d: %s\n", iDev, iProp.name);

  grid.x = 1;
  block.x = 2;

  kernel_checkvalue<<<grid, block>>>();
  cudaDeviceReset();
}
