#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_hello(int nest){
  dim3 grid, block;
  grid.x = 1;
  block.x = 1;
  if(nest < 3){
    printf("Hello DEVICE!!\n");
    printf("nest : %d\n", nest);
    nest++;
    kernel_hello<<<grid, block>>>(nest);
    cudaDeviceSynchronize();
  }
}

void exec(){
  int number;
  int iDev = 0;
  cudaDeviceProp iProp;
  dim3 grid, block;
  grid.x = 1;
  block.x = 1;

  cudaGetDeviceProperties(&iProp, iDev);
  printf("Device %d: %s\n", iDev, iProp.name);
  kernel_hello<<<grid, block>>>(1);
  cudaDeviceSynchronize();
  cudaDeviceReset();
}
int main(int argc, char const *argv[]) {
  exec();
  return 0;
}
