#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include "matrix.h"
#include "caqr.cu"
#define TILE_WIDTH 16
#define Scalar float

void mmqr(Scalar* mat, Scalar* tau, int m, int n);
void getPanelDims(int m, int n, int* rowPanels, int* colPanels);

////////////////////////////////////////////////////////////////////////////////
//  (1) generation of random matrix 𝛺;
//  (2) matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;
//  (3) QR decomposition on 𝑌;
//  (4) matrix-matrix multiplication of 𝑄𝑇𝐴; and
//  (5) deterministic SVD decomposition on 𝐵.
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void find_randomized_range(Matrix M, int rank)
{
 
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Generate Random Matrix alpha or copy from 
    
    // matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;

 
    //  for (int m = 0; m < (int)ceil((float)M.width / blockDim.x); ++m) {
    //      // Collaborative loading of Md and Nd tiles into shared memory
    //      if((m*TILE_WIDTH + tx) < M.width && row < P.height){
    //          Mds[ty][tx] = M.elements[row*M.width + (m*TILE_WIDTH + tx)];
    //      }else
    //      {
    //          Mds[ty][tx] = 0;
    //      }
    //      if((m*TILE_WIDTH + ty) < N.height && col < P.width){
    //          Nds[ty][tx] = N.elements[(m*TILE_WIDTH + ty)* N.width + col];
    //      }else
    //      {
    //          Nds[ty][tx] = 0;
    //      }

    //      __syncthreads();

    //      for(int k = 0; k <TILE_WIDTH; ++k){
    //          Pvalue += Mds[ty][k] * Nds[k][tx];
    //      }
    //      __syncthreads();

    //  }
    //  if(col<P.width && row<P.height){
    //      P.elements[row*P.width + col] = Pvalue;
    //  }
}
 

void qr_decompostion(int m, int n){
  //make m,n fit to panels
  {
    int numPanels = ((double) (m - PR) / (PR - PC) + 0.5);
    m = PR + numPanels * (PR - PC);
  }
  {
    int numPanels = ((double) n / PC + 0.5);
    if(numPanels == 0)
      numPanels = 1;
    n = numPanels * PC;
    while(n > m)
      n -= PC;
  }
  printf("Exact problem size: %dx%d\n", m, n);
  assert(m && n && m >= n);
  //only use one device (at least, for now)
  //First, make sure device is using proper 48 KB of shared, 16 KB L1
  //during all calls to L1 kernel
  //Note that this is not the default
  HANDLE_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int sm = prop.multiProcessorCount;
  printf("Testing mmqr on \"%s\"\n", prop.name);
  printf("Device has %d SMs, %zu bytes of shared, and up to %d threads per block\n", sm, prop.sharedMemPerBlock, prop.maxThreadsPerBlock);
  if(sizeof(Scalar) == 4)
  {
    HANDLE_ERROR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
  }
  else if(sizeof(Scalar) == 8)
  {
    HANDLE_ERROR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  }
  else
  {
    puts("Only float (32-bit) and double (64-bit) reals are supported scalar types");
    exit(1);
  }
  Scalar* A = (Scalar*) malloc(m * n * sizeof(Scalar));
  Scalar* RV = (Scalar*) malloc(m * n * sizeof(Scalar));
  int rowPanels, colPanels;
  getPanelDims(m, n, &rowPanels, &colPanels);
  Scalar* tau = (Scalar*) malloc(rowPanels * colPanels * PC * sizeof(Scalar));
  srand(12);
  //initialize A randomly
  for(int i = 0; i < m * n; i++)
  {
    A[i] = (Scalar) rand() / RAND_MAX;
    RV[i] = A[i];
  }
  //puts("A matrix:\n");
  //printMat(A, m, n);
  double mmqrElapsed = 0;
  struct timeval currentTime;
  gettimeofday(&currentTime, NULL);
  for(int i = 0; i < trials; i++)
  {
    mmqr(RV, tau, m, n);
    struct timeval nextTime;
    gettimeofday(&nextTime, NULL);
    //add to elapsed time
    mmqrElapsed += (nextTime.tv_sec + 1e-6 * nextTime.tv_usec) - (currentTime.tv_sec + 1e-6 * currentTime.tv_usec);
    currentTime = nextTime;
    //refresh RV for next trial (this isn't part of the algorithm and so isn't timed)
    if(i != trials - 1)
      memcpy(RV, A, m * n * sizeof(Scalar));
  }
  printf(" MMQR ran QR on %dx%d matrix in %f s (avg over %d)\n", m, n, mmqrElapsed / trials, trials);
  cudaProfilerStop();
  /*
  printf("tau values after QR (grid corresponding to columns within panels):\n");
  for(int j = 0; j < rowPanels; j++)
  {
    for(int i = 0; i < colPanels * PC; i++)
    {
      printf("%9f ", tau[i * rowPanels + j]);
    }
    putchar('\n');
  }
  putchar('\n');
  */
  //printf("A raw storage after QR:\n");
  //printMat(RV, m, n);
  /*
  Scalar* Q = (Scalar*) malloc(m * m * sizeof(Scalar));
  Scalar* R = (Scalar*) malloc(m * n * sizeof(Scalar));
  explicitQR(RV, tau, Q, R, m, n);
  printf("Q:\n");
  printMat(Q, m, m);
  printf("R:\n");
  printMat(R, m, n);
  //now compute Q*R explicitly and compare to A
  Scalar* QR = (Scalar*) malloc(m * n * sizeof(Scalar));
  dgemm(Q, R, QR, m, m, n);
  printf("QR:\n");
  printMat(QR, m, n);
  Scalar* QRmA = (Scalar*) malloc(m * n * sizeof(Scalar));
  Scalar errNorm = 0;
  for(int i = 0; i < m * n; i++)
  {
    QRmA[i] = QR[i] - A[i];
    errNorm += QRmA[i] * QRmA[i];
  }
  printf("QR-A (should be 0):\n");
  printMat(QRmA, m, n);
  free(QRmA);
  errNorm = sqrt(errNorm);
  printf("L2 norm of residual QR-A: %.9g\n", errNorm);
  free(R);
  free(Q);
  free(QR);
  */
  free(RV);
  free(A);
}

int main(int argc, const char** argv)
{
  HANDLE_ERROR(cudaSetDevice(0));
  if(argc < 3)
  {
    puts("Usage: ./qr_device m n");
    exit(1);
  }
  int m = atoi(argv[1]);
  int n = atoi(argv[2]);

  qr_decompostion(m,n);

  return 0;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
 