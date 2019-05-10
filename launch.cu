#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
// Standard Libs
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include "cuda_runtime.h"
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <stdint.h>

// My Packages
#include "./utils/matrix.h"
#include "./utils/gpuErrorCheck.h"
#include "./utils/rsvd.h"
#include "./utils/math_util_cpu.cpp"
// #include "./kernels/caqr.cu"
// #include "./kernels/matrix_op_kernel.cu"


// CONSTANTS 
#define TILE_WIDTH 16
#define Scalar float

// Function Definitions

////////////////////////////////////////////////////////////////////////////////
//  (1) generation of random matrix 𝛺;
//  (2) matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;
//  (3) QR decomposition on 𝑌;
//  (4) matrix-matrix multiplication of 𝑄𝑇𝐴; and
//  (5) deterministic SVD decomposition on 𝐵.
////////////////////////////////////////////////////////////////////////////////

void rsvd(const uint64_t m, const uint64_t n, const uint64_t k){

    // create cusolverDn/cublas handle
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH) );
    CHECK_CUBLAS( cublasCreate(&cublasH) );

    const uint64_t p = k; // oversampling number
    const uint64_t l = k + p;
    const uint64_t q = 2; // power iteration factor
    assert(l < min(m, n) && "k+p must be < min(m, n)" );

    const uint64_t ldA  = roundup_to_32X( m );  // multiple of 32 by default
    const uint64_t ldVT = roundup_to_32X( l );
    const uint64_t ldU = ldA;

    // allocate device memory
    
    
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

    rsvd(m,n,r);

    return 0;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
 