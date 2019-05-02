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
#include "matrix_op_kernel.cu"
#define TILE_WIDTH 16
#define Scalar float

void mmqr(Scalar* mat, Scalar* tau, int m, int n);
void getPanelDims(int m, int n, int* rowPanels, int* colPanels);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
void explicitQR(Scalar* A, Scalar* tau, Scalar* Q, Scalar* R, int m, int n);

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
 

void qr_decompostion(Scalar* RV, int m, int n){
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
    int rowPanels, colPanels;
    getPanelDims(m, n, &rowPanels, &colPanels);
    Scalar* tau = (Scalar*) malloc(rowPanels * colPanels * PC * sizeof(Scalar));
    srand(12);

    // Apply QR and time it
    double mmqrElapsed = 0;
    struct timeval currentTime;
    gettimeofday(&currentTime, NULL);
    mmqr(RV, tau, m, n);

    

    Scalar* Q = (Scalar*) malloc(m * m * sizeof(Scalar));
    Scalar* R = (Scalar*) malloc(m * n * sizeof(Scalar));
    explicitQR(RV, tau, Q, R, m, n);
    printf("Q:\n");
    printMat(Q, m, m);
    printf("R:\n");
    printMat(R, m, n);
    /*
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

    */
    struct timeval nextTime;
    gettimeofday(&nextTime, NULL);
    mmqrElapsed += (nextTime.tv_sec + 1e-6 * nextTime.tv_usec) - (currentTime.tv_sec + 1e-6 * currentTime.tv_usec);
    currentTime = nextTime;
    printf("Ran QR on %dx%d matrix in %f s\n", m, n, mmqrElapsed);
    cudaProfilerStop();
}

int main(int argc, const char** argv)
{
    if(argc < 4)
    {
        puts("Usage: ./qr_device m n r");
        exit(1);
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int r = atoi(argv[3]);

    //initialize A randomly
    Scalar* A = (Scalar*) malloc(m * n * sizeof(Scalar));
    Scalar* Omega = (Scalar*) malloc(n * r * sizeof(Scalar));
    printf("A:\n");
    for(int i = 0; i < m * n; i++)
    {
        A[i] = (Scalar) rand() / RAND_MAX;
    }
    printf("Omega:\n");
    for(int i = 0; i < n * r; i++)
    {
        Omega[i] = (Scalar) rand() / RAND_MAX;
    }

    printMat(A,5,5);
    printMat(Omega,3,3);
    qr_decompostion(A,m,n);
    free(A);
    free(Omega);
    free(R);
    free(Q);
    free(QR);
    return 0;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
 