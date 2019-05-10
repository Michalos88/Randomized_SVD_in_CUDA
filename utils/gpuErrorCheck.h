#ifndef _ERRORCHECK_H
#define _ERRORCHECK_H
#include <stdio.h>
#include <stdlib.h> // for exit()
#include <cublas.h>
#include <cusolverDn.h>

#define CHECK_CUDA(call)                                                       \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);            \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

static inline char const *cublas_err_string (cublasStatus err){
    switch(err){
        case CUBLAS_STATUS_SUCCESS:         return "Sucess.";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "Not initialized; usually caused by the lack of a prior call or an error in the hardware setup";
        case CUBLAS_STATUS_ALLOC_FAILED:    return "Recousrce allocation falied; usually caused by a cudaMalloc() failure.";
        case CUBLAS_STATUS_INVALID_VALUE:   return "An invalid numerical value was used as an argument(a negative vector size, for example).";
        case CUBLAS_STATUS_ARCH_MISMATCH:   return "An absent device architectural feature is required; usually caused by the lack of support for atomic operations or double precision.";
        case CUBLAS_STATUS_MAPPING_ERROR:   return "An access to GPU memory space failed.";
        case CUBLAS_STATUS_EXECUTION_FAILED:return "The GPU program failed to execute.";
        case CUBLAS_STATUS_INTERNAL_ERROR:  return "An internal operation failed; usually caused by a cudaMemcpyAsync() failure.";
        case CUBLAS_STATUS_NOT_SUPPORTED:   return "The feature required is not supported.";
        default:                            return "Unknown Error Code";
    }
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus err;                                                          \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS){                              \
        fprintf(stderr, "Cublas error at %s:%d -- %s\n", __FILE__, __LINE__,   \
            cublas_err_string (err));                                          \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess){                                          \
            fprintf(stderr, "CUDA ERROR \"%s\" also detected\n",               \
            cudaGetErrorString(cuda_err));                                     \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "CURAND ERROR %d at %s:%d\n", err, __FILE__,           \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "CUFFT ERROR %d at %s:%d\n", err, __FILE__,            \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "CUSPARSE ERROR %d at %s:%d\n", err, __FILE__, __LINE__);\
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "CUDA ERROR \"%s\" also detected\n",               \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

static inline char const *cusolver_err_string (cusolverStatus_t err){
    switch(err){
        case CUSOLVER_STATUS_SUCCESS:         return "Sucess.";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "Not initialized; usually caused by the lack of a prior call or an error in the hardware setup";
        case CUSOLVER_STATUS_ALLOC_FAILED:    return "Recousrce allocation falied; usually caused by a cudaMalloc() failure.";
        case CUSOLVER_STATUS_INVALID_VALUE:   return "An invalid numerical value was used as an argument(a negative vector size, for example).";
        case CUSOLVER_STATUS_ARCH_MISMATCH:   return "An absent device architectural feature is required; usually caused by the lack of support for atomic operations or double precision.";
        case CUSOLVER_STATUS_EXECUTION_FAILED:return "The GPU program failed to execute.";
        case CUSOLVER_STATUS_INTERNAL_ERROR:  return "Internal operation failed; usually caused by a cudaMemcpyAsync() failure.";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:return "The feature required is not supported; usually caused by passing an invalid matrix descriptor to the function.";
        default:                            return "Unknown Error Code";
    }
}


#define CHECK_CUSOLVER(call)                                                   \
{                                                                              \
    cusolverStatus_t err;                                                      \
    if ((err = (call)) != CUSOLVER_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "CUSOLVER ERROR at %s:%d -- %s\n", __FILE__, __LINE__, \
             cusolver_err_string(err));                                        \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "CUDA ERROR \"%s\" also detected\n",               \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_MAGMA( err )                                                   \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "MAGMA Error: %s\nfailed at %s:%d: error %lld: %s\n",\
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

#define CHECK_NCCL(cmd) do {                           \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",         \
    __FILE__,__LINE__,ncclGetErrorString(r));         \
    exit(EXIT_FAILURE);                               \
    }                                                 \
} while(0)

#endif // _ERRORCHECK_H