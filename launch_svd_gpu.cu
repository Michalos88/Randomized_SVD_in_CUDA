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
#include <chrono> // for timer

// My Libs
#include "./utils/matrix.h"
#include "./utils/gpuErrorCheck.h"
#include "./utils/rsvd.h"
#include "./utils/math_util_cpu.cpp"
#include "./kernels/rsvd_device.cu" 

// timer
uint64_t getCurrTime()
{
    return chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void rsvd_on_gpu(const uint64_t m, const uint64_t n, const uint64_t k){

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
        rsvd_gpu(dev_U, dev_S, dev_VT, dev_A, m, n, l, q, cusolverH, cublasH);

        uint64_t tock = getCurrTime();
        double InCoreTime = (tock - tick) / 1e6; //from µs to s
    
        cout << "RSVD On GPU Time: " << InCoreTime << endl;

        double InCoreErr = 0.0;

        CHECK_CUDA( cudaMalloc((void **)&dev_A, ldA * n * sizeof(double)) );
        CHECK_CUDA( cudaMemset(dev_A, 0,   ldA * l * sizeof(double)) );
        CHECK_CUBLAS( cublasSetMatrix(m, n, sizeof(double), host_A, m, dev_A, ldA) );
        InCoreErr = svdFrobeniusDiffGPU(cublasH, dev_A, dev_U, dev_S, dev_VT, m, n, l);
        CHECK_CUDA( cudaFree( dev_A ) );  

        CHECK_CUDA( cudaFree( dev_U ) );
        CHECK_CUDA( cudaFree( dev_S ) );
        CHECK_CUDA( cudaFree( dev_VT) );

        cout << "RSVD Error: " << InCoreErr << endl;

    }else{
        cout << "This Matrix is too big!" << endl;
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
    srand(5672);
    cout << "GPU RSVD Started" << endl;
    rsvd_on_gpu(m,n,r);
    cout << "GPU RSVD Completed" << endl;

    return 0;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
 