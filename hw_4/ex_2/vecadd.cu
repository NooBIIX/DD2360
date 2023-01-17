#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define TPB 1024

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if(id < len) out[id] = in1[id] + in2[id];
}

//@@ Insert code to implement timer start
double timestart(){
  struct timeval t_start;
  gettimeofday(&t_start, NULL);
  return (double) (1000000.0*(t_start.tv_sec) + t_start.tv_usec)/1000000.0;
}
//@@ Insert code to implement timer stop
double timestop(){
  struct timeval t_stop;
  gettimeofday(&t_stop, NULL);
  return (double) (1000000.0*(t_stop.tv_sec) + t_stop.tv_usec)/1000000.0;
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args

  if(argc > 1) inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  size_t bytes = inputLength * sizeof(double);


  hostInput1 = (double *)malloc(bytes);
  hostInput2 = (double *)malloc(bytes);
  hostOutput = (double *)malloc(bytes);
  resultRef = (double *)malloc(bytes);


  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand((unsigned)time(NULL));
  for(int i = 0; i < inputLength - 1; i++){
    hostInput1[i] = ((double)rand()/(double)RAND_MAX);
    hostInput2[i] = ((double)rand()/(double)RAND_MAX);
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, bytes);
  cudaMalloc(&deviceInput2, bytes);
  cudaMalloc(&deviceOutput, bytes);


  //@@ Insert code to below to Copy memory to the GPU here

  double start1 = timestart();

  cudaMemcpy(deviceInput1, hostInput1, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, hostOutput, bytes, cudaMemcpyHostToDevice);

  double t1me = timestop() - start1;
  printf("Transfer time host to device %f seconds\n", t1me);
  //cudaDeviceSynchronize();
  //@@ Initialize the 1D grid and block dimensions here
  int blockSize, gridSize;

  blockSize = TPB;
  gridSize = (inputLength+blockSize-1) / blockSize;
  printf("gridSize is %d\n", gridSize);
  //@@ Launch the GPU Kernel here
  double start_kernel = timestart();

  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();

  double time_kernel = timestop() - start_kernel;
  printf("Kernel time %f seconds\n", time_kernel);

  //@@ Copy the GPU memory back to the CPU here
  double start2 = timestart();

  cudaMemcpy(hostOutput, deviceOutput, bytes, cudaMemcpyDeviceToHost);

  double t2me = timestop() - start2;
  printf("Transfer time device to host %f seconds\n", t2me);
  //@@ Insert code below to compare the output with the reference
  if (hostOutput == resultRef) printf("The output is same as the reference!\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
