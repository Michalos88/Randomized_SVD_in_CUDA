#ifndef _RSVD_H
#define _RSVD_H

#include <cuda_runtime.h>
#include <algorithm> // for max() min()
#include <iostream>
#include <time.h>
#include <cassert>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand_kernel.h> // for random number generator
#include <iomanip>
#include "math_util_gpu.cpp"
#include "math_util_cpu.cpp"
#include <stdint.h>


#define TimerON true
#define testError true


__host__ __device__
static inline uint64_t roundup_to_32X(const uint64_t x){
    return ((x + 32 - 1) / 32) * 32;
}

void rsvd_gpu(double *U, double *S, double *VT, double *A,
              const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
              cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH);

void svd(cusolverDnHandle_t &cusolverH,
         double *d_U, double *d_S, double *d_VT,
         double *d_A, const uint64_t m, const uint64_t n);

void orthogonalization(cusolverDnHandle_t &cusolverH,
                       double *dev_A, const uint64_t m, const uint64_t n);

int orth_CAQR_size(const int m, const int n);

void orth_CAQR(double *d_A, const uint64_t m, const uint64_t n);

void orth_magma(double *d_A, const uint64_t m, const uint64_t n);

void transposeGPU(cublasHandle_t &cublasH,
                  double *A_T, const double *A,
                  const uint64_t M, const uint64_t N);

void genLowRankMatrixGPU(cublasHandle_t &cublasH, double *d_A,
                         const uint64_t M,const uint64_t N,
                         const uint64_t K, const uint64_t ldA);

void inverse_svd(cublasHandle_t &cublasH, double *A,
                 const double *U, const double *S, const double *Vt,
                 const uint64_t m, const uint64_t n,
                 const uint64_t k, const uint64_t ldvt);

double matrixFrobeniusNorm(cublasHandle_t &cublasH, const double *A,
                           const uint64_t m, const uint64_t n, const uint64_t ldA);

double matrixAbsMax(cublasHandle_t &cublasH, const double *A,
                    const uint64_t m, const uint64_t n, const uint64_t ldA);

double FrobeniusNorm(cublasHandle_t &cublasH, const double *A,
                     const uint64_t m, const uint64_t n, const uint64_t ldA);

double matrixL1Norm(cublasHandle_t &cublasH, const double *A,
                    const uint64_t m, const uint64_t n, const uint64_t ldA);

double FrobeniusErr(cublasHandle_t &cublasH, const double *A, const double *B,
                    const uint64_t m, const uint64_t n, const uint64_t ldA, const uint64_t ldB);

double matrixInfinityNorm(cublasHandle_t &cublasH, const double *A,
                          const uint64_t m, const uint64_t n, const uint64_t ldA);

double svdFrobeniusDiffGPU(cublasHandle_t &cublasH, double *A,
                           const double *U, const double *S, const double *Vt,
                           const uint64_t m, const uint64_t n, const uint64_t k);

double svd_l1_err(cublasHandle_t &cublasH, const double *A,
                  const double *U, const double *S, const double *Vt,
                  const uint64_t m, const uint64_t n, const uint64_t k);

void print_device_matrix(const double *dev_A, const uint64_t M, const uint64_t N,
                         const uint64_t lda, const char* name);

void print_column_major_matrix(const double *A, const uint64_t M, const uint64_t N);

void Sewdmm(double *A, const double *B, const uint64_t m, const uint64_t n);

void addNoiseToMatrix(double *A, const uint64_t m, const uint64_t n, const double noise_rate);

void genSparseMatrix(cublasHandle_t &cublasH, double *d_SP,
                     const uint64_t m, const uint64_t n,
                     const uint64_t lda, const double sparsity);

void rsvdOoC(double *host_U, double *host_S, double *host_VT, const double *host_A,
             const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q, const uint64_t s,
             cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH);

void cublasXtRsvd(double *U, double *S, double *VT, double *A,
                  const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                  cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH);

// the following the CPU API
void genNormalRand(double *h_rand, const uint64_t m, uint64_t n);

void genUniformRand(double *h_rand, const uint64_t m, uint64_t n);

void genLowRankMatrix(double *h_A,
                      const uint64_t m, const uint64_t n, const uint64_t k);

double frobeniusNorm(const double *h_A, const uint64_t m, const uint64_t n);

double svdFrobeniusDiff(const double *A,
                        const double *U, const double *S, const double *Vt,
                        const uint64_t m, const uint64_t n, const uint64_t l);


#endif // _RSVD_H