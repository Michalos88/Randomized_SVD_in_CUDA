#include "../utils/gpuErrorCheck.h"
#include "../utils/rsvd.h"
#include "caqr2.cu"
#include "../utils/math_util_gpu.cpp"
using namespace std;

void rsvd_colSampling_gpu(double *U, double *S, double *VT, double *A,
                          const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                          cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    //setup parameters
    const uint64_t ldA     = roundup_to_32X( m ); // pad columns into multiple of 32
    const uint64_t ldOmega = roundup_to_32X( n );
    const uint64_t ldUhat  = roundup_to_32X( l );
    const uint64_t ldU = ldA, ldY = ldA,
    ldBT= ldOmega, ldV = ldOmega, ldP = ldOmega;
    
    // double_one & double_zero for matrix multiplication
    const double double_one = 1.0, double_zero = 0.0;
    
    /*********** Step 1: Y = A * Omega, ************/
    double *Omega;
    CHECK_CUDA(cudaMalloc((void**)&Omega, ldOmega * l * sizeof(double)) );
    curandGenerator_t randGen;
    CHECK_CURAND(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));
    // seeds for curand
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)));
    // generate double normal distribution with mean = 0.0, stddev = 1.0
    CHECK_CURAND(curandGenerateNormalDouble(randGen, Omega, ldOmega * l, 0.0, 1.0));
    CHECK_CURAND(curandDestroyGenerator( randGen ));
    
    double *Y;
    int QR_workSpace = orth_CAQR_size(m, l);
    CHECK_CUDA(cudaMalloc((void**)&Y, QR_workSpace * sizeof(double)) );
    
    // Y = A * Omega, Y[mxl] = A[mxn] * Omega[nxl] memory usage:  m(l+n) + nl
    CHECK_CUBLAS( cublasDgemm( cublasH,  CUBLAS_OP_N, CUBLAS_OP_N,
                              m, l, n,
                              &double_one,
                              A, ldA,
                              Omega, ldOmega,
                              &double_zero,
                              Y, ldY) );
    
    CHECK_CUDA( cudaFree(Omega) );
    
    /********** Step 2: power iteration *********/
    //memory usage:  m(l+n) + nl,
    double *P;
    CHECK_CUDA(cudaMalloc((void**)&P, ldP * l * sizeof(double)) );
    CHECK_CUDA( cudaMemset(P, 0,      ldP * l * sizeof(double)) );
    for(uint64_t i = 0; i < q; i++){
        // P = A' * Y, P[nxl] = A'[nxm] * Y[mxl]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                  n, l, m,
                                  &double_one,
                                  A, ldA,
                                  Y, ldY,
                                  &double_zero,
                                  P, ldP) );
        
        //CHECK_CUDA( cudaMemset(Y, 0, ldA * l * sizeof(double)) );
        
        //Q = A * P, Q[mxl] = A[mxn] * P[nxl], Y is used to save Q
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, n,
                                  &double_one,
                                  A, ldA,
                                  P, ldP,
                                  &double_zero,
                                  Y, ldY) );
        

    }
    
    CHECK_CUDA( cudaFree(P) );
    
    orth_CAQR(Y, m, l);
    
    // orthogonalization(cusolverH, Y, m, l);
    
    /************ Step 3: B' = A' * Q, B'[nxl] = A'[nxm] * Q[mxl] **********/
    // allocate for SVD memory
    double *BT;
    CHECK_CUDA(cudaMalloc((void**)&BT,    ldBT   * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(BT,  0,   ldBT   * l * sizeof(double)) );
    
    
    // Note: becuase cusolver 8.0 can not solve m < n matrix,
    // B is transposed, which different from Halko's algorithm
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                              n, l, m,
                              &double_one,
                              A, ldA,
                              Y, ldY,
                              &double_zero,
                              BT, ldBT) );

    CHECK_CUDA( cudaFree(A) );
    
    //print_device_matrix(BT, n, l, ldBT, "BT");
    
    /********** Step 5: SVD on BT (nxl) *********/
    double *UhatT, *V;
    CHECK_CUDA(cudaMalloc((void**)&UhatT, ldUhat * l * sizeof(double)) );
    CHECK_CUDA(cudaMalloc((void**)&V,     ldV    * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(UhatT,0,  ldUhat * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(V,    0,  ldV    * l * sizeof(double)) );
    
    CHECK_CUDA( cudaThreadSynchronize() );
    
    //V[nxl] * S[lxl] * UhatT[lxl] = BT[nxl]
    svd(cusolverH, V, S, UhatT, BT, n, l);
    
    CHECK_CUDA( cudaFree(BT) );
    
    /********** Step 6:  U = Q * Uhat, U[mxl] = Q[mxl] * Uhat[lxl] *********/
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                              m, l, l,
                              &double_one,
                              Y, ldY,
                              UhatT, ldUhat,
                              &double_zero,
                              U, ldU) );
    
    CHECK_CUDA( cudaFree(UhatT) );
    CHECK_CUDA( cudaFree(Y) );
    
    /**********Step 7: transpose V ****/
    transposeGPU(cublasH, VT, V, n, l);
    CHECK_CUDA( cudaFree(V) );
    
    CHECK_CUDA(cudaMalloc((void**)&A,     sizeof(double) * ldA * n) );

}

void rsvd_rowSampling_gpu(double *U, double *S, double *VT, double *A,
                          const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
                          cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    //setup parameters
    const uint64_t ldA     = roundup_to_32X( m ); // pad columns into multiple of 32
    const uint64_t ldOmega = roundup_to_32X( m );
    
    // double_one & double_zero for matrix multiplication
    const double double_one = 1.0, double_zero = 0.0;

    
    /*********** Step 1: Y = A' * Omega, ************/
    curandGenerator_t randGen;
    CHECK_CURAND(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));
    // seeds for curand
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)));
    // generate double normal distribution with mean = 0.0, stddev = 1.0
    double *Omega;
    CHECK_CUDA(cudaMalloc((void**)&Omega, ldOmega * l * sizeof(double)) );
    CHECK_CURAND(curandGenerateNormalDouble(randGen, Omega, ldOmega * l, 0.0, 1.0));
    CHECK_CURAND(curandDestroyGenerator( randGen ));
    
    const uint64_t ldY = roundup_to_32X( n );
    
    double *Y;
    
    int QR_workSpace = orth_CAQR_size(n, l);
    
    CHECK_CUDA(cudaMalloc((void**)&Y, QR_workSpace * sizeof(double)) );
    //CHECK_CUDA(cudaMalloc((void**)&Y, ldY * l * sizeof(double)) );
    
    // Y = A' * Omega, Y[nxl] = A'[nxm] * Omega[mxl]
    CHECK_CUBLAS( cublasDgemm( cublasH,  CUBLAS_OP_T, CUBLAS_OP_N,
                              n, l, m,
                              &double_one,
                              A, ldA,
                              Omega, ldOmega,
                              &double_zero,
                              Y, ldY) );
    
    CHECK_CUDA( cudaFree(Omega) );
    
    /********** Step 2: power iteration *********/
    const uint64_t ldP = roundup_to_32X( m );
    double *P;
    CHECK_CUDA(cudaMalloc((void**)&P, ldP * l * sizeof(double)) );
    CHECK_CUDA( cudaMemset(P, 0,      ldP * l * sizeof(double)) );
    
    for(uint64_t i = 0; i < q; i++){
        // P =  A * Y, P[mxl] =  A[mxn] * Y[nxl]
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, l, n,
                                  &double_one,
                                  A, ldA,
                                  Y, ldY,
                                  &double_zero,
                                  P, ldP) );
        
        //Q = A' * P, Q[nxl] = A'[nxm] * P[mxl], Y is used to save Q
        CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                  n, l, m,
                                  &double_one,
                                  A, ldA,
                                  P, ldP,
                                  &double_zero,
                                  Y, ldY) );
        

    }
    CHECK_CUDA( cudaFree(P) );
    
    orth_CAQR(Y, n, l);
    //orthogonalization(cusolverH, Y, n, l);
    
    /************ Step 3: B = A * Q **********/
    const uint64_t ldB = roundup_to_32X( m );
    double *B;
    CHECK_CUDA(cudaMalloc((void**)&B,    ldB * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(B,  0,   ldB * l * sizeof(double)) );
    //   B[mxl] = A[mxn] * Q[nxl]
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, l, n,
                              &double_one,
                              A, ldA,
                              Y, ldY,
                              &double_zero,
                              B, ldB) );
    
    CHECK_CUDA( cudaFree(A) );
    
    /********** Step 5: SVD on BT (mxl) *********/
    const uint64_t ldVThat = roundup_to_32X( l );
    double *VThat;
    CHECK_CUDA( cudaMalloc((void**)&VThat, ldVThat * l * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(VThat,  0, ldVThat * l * sizeof(double)) );
    
    CHECK_CUDA( cudaThreadSynchronize() );
    
    //U[mxl] * S[lxl] * VThat[lxl] = BT[mxl]
    svd(cusolverH, U, S, VThat, B, m, l);
    
    CHECK_CUDA( cudaFree(B) );

    
    /********** Step 6:  VT =  VThat * Q', VT[lxn] = VThat[lxl] * Q'[lxn] *********/
    const uint64_t ldVT = roundup_to_32X( l );
    CHECK_CUBLAS( cublasDgemm( cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                              l, n, l,
                              &double_one,
                              VThat, ldVThat,
                              Y, ldY,
                              &double_zero,
                              VT, ldVT) );
    
    CHECK_CUDA( cudaFree(Y) );
    
    // clean up
    CHECK_CUDA( cudaFree(VThat) );
    CHECK_CUDA(cudaMalloc((void**)&A,     sizeof(double) * ldA * n) );
    
}

void rsvd_gpu(double *U, double *S, double *VT, double *A,
              const uint64_t m, const uint64_t n, const uint64_t l, const uint64_t q,
              cusolverDnHandle_t &cusolverH, cublasHandle_t &cublasH){
    
    if(m >= n){
        // column sampling for tall-skinny
        rsvd_colSampling_gpu(U, S, VT, A, m, n, l, q, cusolverH, cublasH);
    
    }else{
        // row sampling for short-wide
        rsvd_rowSampling_gpu(U, S, VT, A, m, n, l, q, cusolverH, cublasH);
    }
    
}