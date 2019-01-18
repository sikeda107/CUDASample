#include <stdio.h>
#include <stdlib.h>

#define N 5
#define BR() printf("\n")
#define BRS(str) printf("%s\n",str)

typedef struct {
  int top;
  int* data;
  int stack_size;
}FIFO;
void exec();
void initialize_array(int*);
void print_array(int*);

int main(int argc, char const *argv[]) {
  exec();
  return 0;
}
// __device__ int i,j,k;

__device__ int push(int new_data,FIFO* stack_t){
  if(stack_t->top > stack_t->stack_size){
    return -1;
  }

  stack_t->data[stack_t->top] = new_data;
  stack_t->top++;
  return 1;
}

__device__ int pop(FIFO* stack_t){

  if(stack_t->top == 0){
    return -1;
  }

  stack_t->top--;
  return 1;
}

__device__ int initialize_stack(FIFO* stack_t,int stack_size){

  stack_t->top = 0;
  stack_t->stack_size = stack_size;

  stack_t->data = (int*) malloc(stack_size*sizeof(int));
  if(stack_t->data == NULL){
    return -1;
  }

  return 1;
}

__device__ int top(FIFO* stack_t){
  if(stack_t->top == 0){
    return -1;
  }
  return stack_t->data[stack_t->top-1];
}

__device__ int isEmpty(FIFO* stack_t){
  if(stack_t->top == 0)
  return 1;
  else
  return 0;
}


__device__ void  swap(int *x, int *y)
{
  int  tmp;
  tmp = *x;
  *x = *y;
  *y = tmp;
}

__device__ void print_d_array(int *array){
  int i;
  BRS(__func__);
  printf("blockIdx.x %d , threadIdx.x %d\n", blockIdx.x, threadIdx.x);
  for (i = 0; i < N; i++) {
    printf("%d ",array[i]);
  }//for
  BR();
}
__global__ void kernel_test_stack(int *d_array){
  int status;
  int i, x = 3, y = 6;
  FIFO stack1;

  print_d_array(d_array);

  //スワップの確認
  printf("x: %d y: %d\n", x, y);
  swap(&x,&y);
  printf("x: %d y: %d\n", x, y);

  //スタックの確認
  if ((status = initialize_stack(&stack1, N)) == -1) {
    printf("initialize_stack error LINE:%d \n", __LINE__);
  }

  printf("blockIdx.x %d , threadIdx.x %d stack address %p x %p y%p \n", blockIdx.x, threadIdx.x, &stack1, &x, &y);
  if(isEmpty(&stack1)){
    BRS("Empty");
  }//if
  else{
    BRS("NOT Empty");
  }//else

  for(i = 1 ; i < N ; i++){
    push(i, &stack1);
    printf("push: %d\n",i);
    if(isEmpty(&stack1)){
      BRS("Empty");
      // printf("top: %d \n",top(&stack1));
    }//if
    else{
      BRS("NOT Empty");
      // printf("top: %d \n",top(&stack1));
    }//else
  }//for
  for(i = 1 ; i < N ; i++){
    pop(&stack1);
    BRS("pop");
    if(isEmpty(&stack1)){
      BRS("Empty");
      printf("top: %d \n",top(&stack1));
    }//if
    else{
      BRS("NOT Empty");
      printf("top: %d \n",top(&stack1));
    }//else
  }//for
}//Kernel

void exec(){
  int array[N];
  int *d_array;
  int iDev = 0;
  dim3 grid, block;
  cudaDeviceProp iProp;
  cudaSetDevice(iDev);
  cudaGetDeviceProperties(&iProp, iDev);
  printf("Device %d: %s\n", iDev, iProp.name);

  initialize_array(array);
  print_array(array);

  cudaMalloc((int**)&d_array, sizeof(array));
  cudaMemcpy(d_array, array, sizeof(array), cudaMemcpyHostToDevice);

  grid.x = 1;
  block.x = 2;

  kernel_test_stack<<<grid, block>>>(d_array);

  cudaMemcpy(array, d_array, sizeof(array), cudaMemcpyDeviceToHost);
  print_array(array);
  cudaFree(d_array);
  cudaDeviceReset();
}

void initialize_array(int* array){
  int i;
  for (i = 0; i < N; i++) {
    array[i] = rand() % N * 2;
  }//for
}//function

void print_array(int* array){
  int i;
  BRS(__func__);
  for (i = 0; i < N; i++) {
    printf("%d ",array[i]);
  }//for
  BR();
}//function
