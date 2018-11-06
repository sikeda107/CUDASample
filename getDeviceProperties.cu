#include "./common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

int samples(void);
int test(void);

int main(int argc, char const *argv[]) {
  samples();
  test();
  return 0;
}
/*
* Fetches basic information on the first device in the current CUDA platform,
* including number of SMs, bytes of constant memory, bytes of shared memory per
* block, etc.
*/

int samples()
{
  int iDev = 0;
  cudaDeviceProp iProp;
  CHECK(cudaGetDeviceProperties(&iProp, iDev));

  printf("Device %d: %s\n", iDev, iProp.name);
  printf("  Number of multiprocessors:                     %d\n",
  iProp.multiProcessorCount);
  printf("  Total amount of constant memory:               %4.2f KB\n",
  iProp.totalConstMem / 1024.0);
  printf("  Total amount of shared memory per block:       %4.2f KB\n",
  iProp.sharedMemPerBlock / 1024.0);
  printf("  Total number of registers available per block: %d\n",
  iProp.regsPerBlock);
  printf("  Warp size:                                     %d\n",
  iProp.warpSize);
  printf("  Maximum number of threads per block:           %d\n",
  iProp.maxThreadsPerBlock);
  printf("  Maximum number of threads per multiprocessor:  %d\n",
  iProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of warps per multiprocessor:    %d\n",
  iProp.maxThreadsPerMultiProcessor / 32);
  printf(" compute capability : %d.%d\n", iProp.major, iProp.minor);
  return EXIT_SUCCESS;
}

// reference URL:
// http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%A5%C7%A5%D0%A5%A4%A5%B9%BE%F0%CA%F3%A4%CE%BC%E8%C6%C0

int test(){

  int n;    //デバイス数
  // cutilSafeCall(cudaGetDeviceCount(&n));
  CHECK(cudaGetDeviceCount(&n));
  for(int i = 0; i < n; ++i){
    cudaDeviceProp dev;

    // デバイスプロパティ取得
    CHECK(cudaGetDeviceProperties(&dev, i));

    printf("device %d\n", i);
    printf(" device name : %s\n", dev.name);
    printf(" total global memory : %d (MB)\n", dev.totalGlobalMem/1024/1024);
    printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);
    printf(" register / block : %d\n", dev.regsPerBlock);
    printf(" warp size : %d\n", dev.warpSize);
    printf(" max pitch : %d (B)\n", dev.memPitch);
    printf(" max threads / block : %d\n", dev.maxThreadsPerBlock);
    printf(" max size of each dim. of block : (%d, %d, %d)\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
    printf(" max size of each dim. of grid  : (%d, %d, %d)\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
    printf(" clock rate : %d (MHz)\n", dev.clockRate/1000);
    printf(" total constant memory : %d (KB)\n", dev.totalConstMem/1024);
    printf(" compute capability : %d.%d\n", dev.major, dev.minor);
    printf(" alignment requirement for texture : %d\n", dev.textureAlignment);
    printf(" device overlap : %s\n", (dev.deviceOverlap ? "ok" : "not"));
    printf(" num. of multiprocessors : %d\n", dev.multiProcessorCount);
    printf(" kernel execution timeout : %s\n", (dev.kernelExecTimeoutEnabled ? "on" : "off"));
    printf(" integrated : %s\n", (dev.integrated ? "on" : "off"));
    printf(" host memory mapping : %s\n", (dev.canMapHostMemory ? "on" : "off"));

    printf(" compute mode : ");
    if(dev.computeMode == cudaComputeModeDefault) printf("default mode (multiple threads can use) \n");
    else if(dev.computeMode == cudaComputeModeExclusive) printf("exclusive mode (only one thread will be able to use)\n");
    else if(dev.computeMode == cudaComputeModeProhibited) printf("prohibited mode (no threads can use)\n");
  }
  return EXIT_SUCCESS;
}
