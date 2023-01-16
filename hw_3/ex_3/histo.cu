
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 1024
#define TPB1 32
#define TPB2 32

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ unsigned int shared_bins[NUM_BINS];

   if(threadIdx.x == 0){
      for (int i = 0; i < NUM_BINS; i++)
          shared_bins[i] = 0;
    }

  __syncthreads();

    atomicAdd(&shared_bins[input[idx]], 1);

  __syncthreads();

  if(threadIdx.x == 0){
    for (int i = 0; i < NUM_BINS; i++){
      atomicAdd(&bins[i],shared_bins[i]);
    }
  }

}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127

  int bins_id = blockDim.x * blockIdx.x + threadIdx.x;

  if(bins_id > num_bins) return;

  if(bins[bins_id] > 127) bins[bins_id] = 127;

}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args

  if(argc > 1) inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output

  hostBins = (unsigned int*)malloc(sizeof(unsigned int) * NUM_BINS);
  hostInput = (unsigned int*)malloc(sizeof(unsigned int) * inputLength);
  resultRef = (unsigned int*)malloc(sizeof(unsigned int) * NUM_BINS);
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)

  srand((unsigned)time(NULL));
  for (int i = 0; i < inputLength; i++){
    hostInput[i] = (unsigned int)rand() % (unsigned int)rand();
  }

  //@@ Insert code below to create reference result in CPU

  for (int i=0;i<inputLength;i++){
    resultRef[i] = 0;
  }

  for (int i=0;i<inputLength;i++){
    resultRef[hostInput[i]]=resultRef[hostInput[i]]+1;
  }

  for (int i=0;i<NUM_BINS;i++){
    if(resultRef[hostInput[i]]>127)
      resultRef[hostInput[i]]=127;
  }

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceInput, sizeof(unsigned int) * inputLength);
  cudaMalloc(&deviceBins, sizeof(unsigned int) * NUM_BINS);

  //@@ Insert code to Copy memory to the GPU here

  cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  
  cudaMemset(deviceBins, 0, sizeof(unsigned int) * NUM_BINS);

  //@@ Initialize the grid and block dimensions here

  dim3 dimBlock1(TPB1,1,1);
  dim3 dimGrid1((inputLength+TPB1-1)/TPB1,1,1);

  //@@ Launch the GPU Kernel here

  histogram_kernel<<<dimGrid1,dimBlock1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  //@@ Initialize the second grid and block dimensions here

  dim3 dimBlock2(TPB2,1,1);
  dim3 dimGrid2((inputLength+TPB2-1)/TPB2,1,1);
  //@@ Launch the second GPU Kernel here

  convert_kernel<<<dimGrid2,dimBlock2>>>(deviceBins,NUM_BINS);
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);
  //@@ Insert code below to compare the output with the reference

  for (int i=0;i<NUM_BINS;i++){
    printf("%d, ",hostBins[i]);
  }

  printf("END!\n");

  for (int i=0;i<NUM_BINS;i++){
    printf("%d, ",resultRef[i]);
  }

  if (hostBins == resultRef) printf("The output is same as the reference!\n");  
  //@@ Free the GPU memory here

  cudaFree(deviceInput);
  cudaFree(deviceBins);
  //@@ Free the CPU memory here

  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

