
#include <stdio.h>
#include <sys/time.h>

#define DataType double
#define TPB 32

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  int row_id = blockIdx.y * blockDim.y + threadIdx.y;

  if ((col_id<numBColumns)&&(row_id<numARows)){
    DataType tmp = 0;
    for (int i=0;i<numAColumns;i++){
      tmp += A[numAColumns*row_id+i] * B[numBColumns*i+col_id];
    }
    C[numBColumns*row_id+col_id] = tmp;
  }
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
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output

  hostA = (DataType*) malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType*) malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  srand((unsigned)time(NULL));
  for(int i=0;i<numARows;i++){
    for (int j=0;j<numAColumns;j++){
      hostA[i*numAColumns+j] = (DataType)rand() / (DataType)RAND_MAX;
    }
  }

    for(int i=0;i<numBRows;i++){
    for (int j=0;j<numBColumns;j++){
      hostB[i*numBColumns+j] = (DataType)rand() / (DataType)RAND_MAX;
    }
  }

  for(int i=0;i<numARows;i++) {
        for(int j=0;j<numBColumns;j++) {
          resultRef[i*numBColumns+j] = 0;
          for(int k=0;k<numAColumns;k++) {
            resultRef[i*numBColumns+j] += hostA[i*numAColumns+k] * hostB[k*numBColumns+j];
          }        
        }
  }

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here

  double start1 = timestart();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double time1 = timestop() - start1;
  printf("Transfer time host to device %f seconds\n", time1);

  //@@ Initialize the grid and block dimensions here

  int dimBlockx= TPB;
  int dimBlocky= TPB;

  int dimGridx= (numCColumns+dimBlockx-1)/dimBlockx;
  int dimGridy= (numCRows+dimBlocky-1)/dimBlocky;

  //@@ Launch the GPU Kernel here
  double start2 = timestart();
  gemm<<<dim3(dimGridx,dimGridy,1),dim3(dimBlockx,dimBlocky,1)>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns);
  cudaDeviceSynchronize();
  double time2 = timestop() - start2;
  printf("Kernel Time %f seconds\n", time2);

  //@@ Copy the GPU memory back to the CPU here
  
  double start3=timestart();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  double time3 = timestop() - start3;
  printf("Transfer Time device to host %f seconds\n", time3);

  //@@ Insert code below to compare the output with the reference
 if (hostC == resultRef) printf("The output is same as the reference!\n");

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}
