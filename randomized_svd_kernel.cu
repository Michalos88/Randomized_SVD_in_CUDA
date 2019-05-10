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
#include "./utils/matrix.h"
// #include "./kernels/caqr.cu"
// #include "./kernels/matrix_op_kernel.cu"
#include "./utils/gpuErrorCheck.h"
#include "./utils/rsvd.h"
#include "./utils/math_util_gpu.cpp"
#include "./utils/math_util_cpu.cpp"
#define TILE_WIDTH 16
#define Scalar float

#include "cuda_runtime.h"
// #include "device_launch_paraMeters.h"

#include<iostream>
#include<iomanip>
#include<cusolverDn.h>
#include<cuda_runtime_api.h>

// #include "Utilities.cuh"

// void mmqr(Scalar* mat, Scalar* tau, int m, int n);
// void getPanelDims(int m, int n, int* rowPanels, int* colPanels);
// void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
// void explicitQR(Scalar* A, Scalar* tau, Scalar* Q, Scalar* R, int m, int n);


// Matrix multiplication kernel thread specification

// __global__ void find_randomized_range(Matrix M, int rank)
// {
 
//     int bx = blockIdx.x; int by = blockIdx.y;
//     int tx = threadIdx.x; int ty = threadIdx.y;

//     // Identify the row and column of the Pd element to work on
//     int row = by * TILE_WIDTH + ty;
//     int col = bx * TILE_WIDTH + tx;

//     // Generate Random Matrix alpha or copy from 
    
//     // matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;

 
//     //  for (int m = 0; m < (int)ceil((float)M.width / blockDim.x); ++m) {
//     //      // Collaborative loading of Md and Nd tiles into shared memory
//     //      if((m*TILE_WIDTH + tx) < M.width && row < P.height){
//     //          Mds[ty][tx] = M.elements[row*M.width + (m*TILE_WIDTH + tx)];
//     //      }else
//     //      {
//     //          Mds[ty][tx] = 0;
//     //      }
//     //      if((m*TILE_WIDTH + ty) < N.height && col < P.width){
//     //          Nds[ty][tx] = N.elements[(m*TILE_WIDTH + ty)* N.width + col];
//     //      }else
//     //      {
//     //          Nds[ty][tx] = 0;
//     //      }

//     //      __syncthreads();

//     //      for(int k = 0; k <TILE_WIDTH; ++k){
//     //          Pvalue += Mds[ty][k] * Nds[k][tx];
//     //      }
//     //      __syncthreads();

//     //  }
//     //  if(col<P.width && row<P.height){
//     //      P.elements[row*P.width + col] = Pvalue;
//     //  }
// }
 

// void qr_decompostion(Scalar* RV, int m, int n){
//     //make m,n fit to panels
//     // {
//     //     int numPanels = ((double) (m - PR) / (PR - PC) + 0.5);
//     //     m = PR + numPanels * (PR - PC);
//     // }
//     // {
//     //     int numPanels = ((double) n / PC + 0.5);
//     //     if(numPanels == 0)
//     //     numPanels = 1;
//     //     n = numPanels * PC;
//     //     while(n > m)
//     //     n -= PC;
//     // }
    
//     // printf("Exact problem size: %dx%d\n", m, n);
//     // assert(m && n && m >= n);

//     //only use one device (at least, for now)
//     //First, make sure device is using proper 48 KB of shared, 16 KB L1
//     //during all calls to L1 kernel
//     //Note that this is not the default
//     HANDLE_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
//     cudaDeviceProp prop;
//     HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
//     int sm = prop.multiProcessorCount;
//     printf("Testing mmqr on \"%s\"\n", prop.name);
//     printf("Device has %d SMs, %zu bytes of shared, and up to %d threads per block\n", sm, prop.sharedMemPerBlock, prop.maxThreadsPerBlock);
//     if(sizeof(Scalar) == 4)
//     {
//         HANDLE_ERROR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
//     }
//     else if(sizeof(Scalar) == 8)
//     {
//         HANDLE_ERROR(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
//     }
//     else
//     {
//         puts("Only float (32-bit) and double (64-bit) reals are supported scalar types");
//         exit(1);
//     }
//     int rowPanels, colPanels;
//     getPanelDims(m, n, &rowPanels, &colPanels);
//     Scalar* tau = (Scalar*) malloc(rowPanels * colPanels * PC * sizeof(Scalar));
//     srand(12);

//     // Apply QR and time it
//     double mmqrElapsed = 0;
//     struct timeval currentTime;
//     gettimeofday(&currentTime, NULL);
//     printMat(RV, 3, 3);
//     mmqr(RV, tau, m, n);
//     printMat(RV, 3, 3);

//     // Scalar* Q = (Scalar*) malloc(m * m * sizeof(Scalar));
//     // Scalar* R = (Scalar*) malloc(m * n * sizeof(Scalar));
//     // explicitQR(RV, tau, Q, R, m, n);
//     // printf("Q:\n");
//     // printMat(Q, m, m);
//     // printf("R:\n");
//     // printMat(R, m, n);
//     /*
//     //now compute Q*R explicitly and compare to A
//     Scalar* QR = (Scalar*) malloc(m * n * sizeof(Scalar));
//     dgemm(Q, R, QR, m, m, n);
//     printf("QR:\n");
//     printMat(QR, m, n);
//     Scalar* QRmA = (Scalar*) malloc(m * n * sizeof(Scalar));
//     Scalar errNorm = 0;
//     for(int i = 0; i < m * n; i++)
//     {
//         QRmA[i] = QR[i] - A[i];
//         errNorm += QRmA[i] * QRmA[i];
//     }
//     printf("QR-A (should be 0):\n");
//     printMat(QRmA, m, n);
//     free(QRmA);
//     errNorm = sqrt(errNorm);
//     printf("L2 norm of residual QR-A: %.9g\n", errNorm);

//     */
//     // struct timeval nextTime;
//     // gettimeofday(&nextTime, NULL);
//     // mmqrElapsed += (nextTime.tv_sec + 1e-6 * nextTime.tv_usec) - (currentTime.tv_sec + 1e-6 * currentTime.tv_usec);
//     // currentTime = nextTime;
//     // printf("Ran QR on %dx%d matrix in %f s\n", m, n, mmqrElapsed);
//     // cudaProfilerStop();
//     free(RV);
//     // free(Q);
//     // free(R);

// }

////////////////////////////////////////////////////////////////////////////////
//  (1) generation of random matrix 𝛺;
//  (2) matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;
//  (3) QR decomposition on 𝑌;
//  (4) matrix-matrix multiplication of 𝑄𝑇𝐴; and
//  (5) deterministic SVD decomposition on 𝐵.
////////////////////////////////////////////////////////////////////////////////

void rsvd(const uint64_t m, const uint64_t n, const uint64_t r){

    const uint64_t p = k; // oversampling number
    const uint64_t l = k + p;

    assert(l < min(m, n) && "k+p must be < min(m, n)" );

    const uint64_t ldA  = roundup_to_32X( m );  // multiple of 32 by default
    const uint64_t ldVT = roundup_to_32X( l );
    const uint64_t ldU = ldA;
    
    // allocate host memory as pinned memory
    double *host_S1;

    CHECK_CUDA( cudaHostAlloc( (void**)&host_S1,     l * sizeof(double), cudaHostAllocPortable ) );
    
    double *host_A, *host_U, *host_S, *host_VT;
    CHECK_CUDA( cudaMallocHost((void**)&host_A, m * n * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&host_U, m * l * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&host_S,     l * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&host_VT,l * n * sizeof(double)) );
    
    /* generate random low rank matrix A ***/
    //genLowRankMatrixGPU(cublasH, dev_A, m, n, k, ldA);
    genLowRankMatrix(host_A, m, n, k);

    size_t freeMem, totalMem;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    double dataSize = m * n * sizeof(double);

    /*********************************** In-core rSVD ***********************************/
    double *dev_A, *dev_U, *dev_S, *dev_VT;
    uint64_t tick = getCurrTime();

    if(dataSize < freeMem * 0.7){
        CHECK_CUDA( cudaMalloc((void **)&dev_A, ldA * n * sizeof(double)) );
        CHECK_CUDA( cudaMalloc((void **)&dev_U, ldU * l * sizeof(double)) );
        CHECK_CUDA( cudaMalloc((void **)&dev_S,       l * sizeof(double)) );
        CHECK_CUDA( cudaMalloc((void **)&dev_VT,ldVT* n * sizeof(double)) );
        CHECK_CUDA( cudaMemsetAsync(dev_A, 0,   ldA * l * sizeof(double)) );
        CHECK_CUDA( cudaMemsetAsync(dev_U, 0,   ldU * l * sizeof(double)) );
        CHECK_CUDA( cudaMemsetAsync(dev_S, 0,         l * sizeof(double)) );
        CHECK_CUDA( cudaMemsetAsync(dev_VT,0,   ldVT* n * sizeof(double)) );
        CHECK_CUBLAS( cublasSetMatrix(m, n, sizeof(double), host_A, m, dev_A, ldA) );
        
        //print_device_matrix(dev_A, m, n, ldA, "A");
        
        rsvd_gpu(dev_U, dev_S, dev_VT, dev_A, m, n, l, q, cusolverH, cublasH);
    }
    uint64_t tock = getCurrTime();
    double InCoreTime = (tock - tick) / 1e6; //from µs to s
    
    double InCoreErr = 0.0;
    
    if( dataSize < freeMem * 0.7 && testError == true){
        
        CHECK_CUDA( cudaMalloc((void **)&dev_A, ldA * n * sizeof(double)) );
        CHECK_CUDA( cudaMemset(dev_A, 0,   ldA * l * sizeof(double)) );
        CHECK_CUBLAS( cublasSetMatrix(m, n, sizeof(double), host_A, m, dev_A, ldA) );
        InCoreErr = svdFrobeniusDiffGPU(cublasH, dev_A, dev_U, dev_S, dev_VT, m, n, l);
        CHECK_CUDA( cudaFree( dev_A ) );        
    }
    if( dataSize < freeMem * 0.7){
        // clean up memory

        CHECK_CUDA( cudaFree( dev_U ) );
        CHECK_CUDA( cudaFree( dev_S ) );
        CHECK_CUDA( cudaFree( dev_VT) );
    }
}





int main(int argc, const char** argv)
{
    if(argc < 4)
    {
        puts("Please Provide: m n r");
        exit(1);
    }
    uint64_t m = atoi(argv[1]);
    uint64_t n = atoi(argv[2]);
    uint64_t r = atoi(argv[3]); // target rank

    //initialize CUDA - solver 
    // --- CUDA solver initialization
	// cusolverDnHandle_t solver_handle;
	// cusolverDnCreate(&solver_handle);


    // //initialize A randomly
    // Scalar* A = (Scalar*) malloc(m * n * sizeof(Scalar));
    // Scalar* Omega = (Scalar*) malloc(n * r * sizeof(Scalar));
    // printf("A:\n");
    // for(int i = 0; i < m * n; i++)
    // {
    //     A[i] = (Scalar) rand() / RAND_MAX;
    // }
    // printf("Omega:\n");
    // for(int i = 0; i < n * r; i++)
    // {
    //     Omega[i] = (Scalar) rand() / RAND_MAX;
    // }
    rsvd(m,n,r);
    // printMat(A,5,5);
    // printMat(Omega,3,3);
    // qr_decompostion(A,m,n);
    // free(A);
    // free(Omega);

    return 0;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
 