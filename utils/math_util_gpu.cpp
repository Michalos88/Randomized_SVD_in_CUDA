#include "gpuErrorCheck.h"
#include "rsvd.h"

using namespace std;

void genLowRankMatrixGPU(cublasHandle_t &cublasH, double *d_A,
                         const uint64_t M, const uint64_t N,
                         const uint64_t K, const uint64_t ldA){
    
    curandGenerator_t randGen;
    
    CHECK_CURAND( curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT) );
    
    // seed
    CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)) );
    
    double *d_rand1, *d_rand2;
    CHECK_CUDA( cudaMalloc((void**)&d_rand1, M * K * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**)&d_rand2, K * N * sizeof(double)) );
    
    
    // set the d_A to 0
    CHECK_CUDA( cudaMemsetAsync(d_A, 0, ldA * N * sizeof(double)) );
    
    // generate double Normal distribution with mean = 0, stddev = 1
    CHECK_CURAND( curandGenerateNormalDouble(randGen, d_rand1, M * K, 0.0f, 1.0f) );
    CHECK_CURAND( curandGenerateNormalDouble(randGen, d_rand2, K * N, 0.0f, 1.0f) );
    
    
    // data = rand1 * rand2, data:[M, N],  rand1:[M, K], rand2:[K, N]
    const double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS( cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K,
                              &alpha,
                              d_rand1,   M,
                              d_rand2,   K,
                              &beta,
                              d_A, ldA) );
    
    CHECK_CURAND( curandDestroyGenerator(randGen) );
    CHECK_CUDA( cudaFree(d_rand1) );
    CHECK_CUDA( cudaFree(d_rand2) );
    
}

void inverse_svd(cublasHandle_t &cublasH, double *A,
                 const double *U, const double *S, const double *Vt,
                 const uint64_t m, const uint64_t n,
                 const uint64_t k, const uint64_t ldvt){
    
    const uint64_t ldA = roundup_to_32X( m );
    //const uint64_t ldvt =roundup_to_32X( n );
    const uint64_t ldw = roundup_to_32X( k );
    
    double *W; // W = S * Vt
    CHECK_CUDA( cudaMalloc ((void**)&W, sizeof(double) * ldw * n) );
    
    CHECK_CUDA( cudaMemset(W, 0, ldw * n * sizeof(double)) );
    CHECK_CUDA( cudaMemset(A, 0, ldA * n * sizeof(double)) );
    
    // S is transformed into a diagonal matrix
    // W = S * Vt,  W[k, n] = S[k, k] * Vt[k, n]
    CHECK_CUBLAS( cublasDdgmm( cublasH, CUBLAS_SIDE_LEFT,
                              k, n,
                              Vt, ldvt,
                              S, 1,
                              W, ldw) );
    
    // A := U * W, A[m, n] =  U[m, k] * W[k, n]
    const double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS( cublasDgemm_v2( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 U, ldA,
                                 W, ldw,
                                 &beta,
                                 A, ldA) );
    
    CHECK_CUDA( cudaFree(W) );
    
};

/* matrix-matrix Frobenius norm  will be calculated as
 * error = |A-B|_f / |A|_f
 */
double FrobeniusErr(cublasHandle_t &cublasH, const double *A, const double *B,
                   const uint64_t m, const uint64_t n,
                   const uint64_t ldA, const uint64_t ldB){
    double *d_TEMP;
    CHECK_CUDA( cudaMalloc ((void**)&d_TEMP, m * n * sizeof(double)) );
    CHECK_CUDA( cudaMemset(d_TEMP,    0,     m * n * sizeof(double)) );
    
    // copy A to d_TEMP to remove padding
    const double f_one = 1.0, f_zero = 0.0;
    CHECK_CUBLAS( cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, // output dimension
                              &f_one,  A,   ldA,
                              &f_zero, NULL,ldA,
                              d_TEMP, m ) );
    
    double frobenNorm_A = 0.0;
    CHECK_CUBLAS( cublasDnrm2(cublasH,  m * n, d_TEMP, 1, &frobenNorm_A) );
    
    // TEMP = A - B
    const double f_minus_one = -1.0;
    CHECK_CUBLAS( cublasDgeam( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n,
                              &f_one,
                              A, ldA,
                              &f_minus_one,
                              B, ldB,
                              d_TEMP, m) );

    //print_device_matrix(d_TEMP, m, n, m, "TEMP=");
    double frobenNorm_DIFF = 0.0;
    CHECK_CUBLAS( cublasDnrm2(cublasH,  m * n, d_TEMP, 1, &frobenNorm_DIFF) );

    //printf("A-B = %0.3e\n", frobenNorm_DIFF);
    
    CHECK_CUDA( cudaFree(d_TEMP) );
    return frobenNorm_DIFF / frobenNorm_A;
    
}

//// WARNING: make sure the padding of input matrix are all set to 0!!!!
//double FrobeniusNorm(cublasHandle_t &cublasH, const double *A,
//                     const uint64_t m, const uint64_t n, const uint64_t ldA){
//
//    double frobenNorm_A = 0.0;
//    CHECK_CUBLAS( cublasDnrm2(cublasH,  ldA * n, A, 1, &frobenNorm_A) );
//    return frobenNorm_A;
//    
//}

double FrobeniusNorm(cublasHandle_t &cublasH, const double *A,
                    const uint64_t m, const uint64_t n, const uint64_t ldA){
    double *d_TEMP;
    CHECK_CUDA( cudaMalloc ((void**)&d_TEMP, m * n * sizeof(double)) );
    
    // copy A to d_TEMP to remove padding
    const double double_one = 1.0, double_zero = 0.0;
    CHECK_CUBLAS( cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, // output dimension
                              &double_one,  A,   ldA,
                              &double_zero, NULL,ldA,
                              d_TEMP, m) );
    
    double frobenNorm_A = 0.0;
    CHECK_CUBLAS( cublasDnrm2(cublasH,  m * n, d_TEMP, 1, &frobenNorm_A) );
    
    CHECK_CUDA( cudaFree(d_TEMP) );
    return frobenNorm_A;
    
}

// get the absolute maxium of a matrix
double matrixAbsMax(cublasHandle_t &cublasH, const double *A,
                    const uint64_t m, const uint64_t n, const uint64_t ldA){
    double *d_TEMP;
    CHECK_CUDA( cudaMalloc( (void**)&d_TEMP, m * n * sizeof(double)) );
    // copy A to d_TEMP to remove padding
    const double f_one = 1.0, f_zero = 0.0;
    CHECK_CUBLAS( cublasDgeam( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, // output dimension
                              &f_one,  A,   ldA,
                              &f_zero, NULL,ldA,
                              d_TEMP, m ) );
    int maxIndex = 0;
    CHECK_CUBLAS( cublasIdamax( cublasH, m * n, d_TEMP, 1, &maxIndex) );
    double host_max = 0.0;
    CHECK_CUDA( cudaMemcpy(&host_max, d_TEMP + maxIndex - 1,// -1 is VERY IMPORTANT
                           sizeof(double), cudaMemcpyDeviceToHost ) );
    
    CHECK_CUDA( cudaFree(d_TEMP) );
    return fabs(host_max);
}

double matrixL1Norm(cublasHandle_t &cublasH, const double *A,
                   const uint64_t m, const uint64_t n, const uint64_t ldA){
    
    double *d_TEMP;
    CHECK_CUDA( cudaMalloc ((void**)&d_TEMP, m * n * sizeof(double)) );
    
    // copy A to d_TEMP to remove padding
    const double f_one = 1.0, f_zero = 0.0;
    CHECK_CUBLAS( cublasDgeam( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, // output dimension
                              &f_one,  A,   ldA,
                              &f_zero, NULL,ldA,
                              d_TEMP, m ) );
    
    double l1Norm = 0.0;
    CHECK_CUBLAS( cublasDasum(cublasH, m * n, d_TEMP, 1, &l1Norm) );
    
    CHECK_CUDA( cudaFree(d_TEMP) );
    
    return l1Norm;
    
}

/*A will be overwritten! */
double svdFrobeniusDiffGPU(cublasHandle_t &cublasH, double *A,
                           const double *U, const double *S, const double *Vt,
                           const uint64_t m, const uint64_t n, const uint64_t l){
    
    const uint64_t ldA = roundup_to_32X( m );
    const uint64_t ldS = roundup_to_32X( l );
    const uint64_t ldvt= roundup_to_32X( l );
    
    double *d_TEMP;
    CHECK_CUDA( cudaMalloc((void**)&d_TEMP, ldS * n * sizeof(double)) );
    //CHECK_CUDA( cudaMemset(d_TEMP, 0,   ldS * n * sizeof(double)) );
    
    double frobenNorm_A = 0.0;
    CHECK_CUBLAS( cublasDnrm2(cublasH,  ldA * n, A, 1, &frobenNorm_A) );
    
    // S is transformed into a diagonal matrix
    // d_TEMP = S * VT,  d_TEMP[l, n] = S[l, l] * Vt[l, n]
    CHECK_CUBLAS( cublasDdgmm( cublasH, CUBLAS_SIDE_LEFT,
                              l, n,
                              Vt,  ldvt,
                              S,   1,   //careful S is vector
                              d_TEMP, ldS) );
    
    // print_device_matrix(SVt, k, n, ldS, "S*Vt");
    
    // DIFF := -U * d_TEMP + A, DIFF[m, n] = -1 * U[m, l] * d_TEMP[l, n] + A[m, n]
    const double double_minus_one = -1.0, double_one = 1.0;
    CHECK_CUBLAS( cublasDgemm_v2( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, l,
                                 &double_minus_one,
                                 U,    ldA,
                                 d_TEMP, ldS,
                                 &double_one,
                                 A, ldA) );
    
    double frobenNorm_DIFF = 0.0;
    // cublasDnrm2 calculate the euclidean norm of a vector
    // the Frobenius Norm of d_A is calculated by treating d_A as a vector
    CHECK_CUBLAS( cublasDnrm2(cublasH,  ldA * n, A, 1, &frobenNorm_DIFF) );
    
    // absolute sum of the matrix
    //CHECK_CUBLAS( cublasDasum(cublasH, m * n, L1NORM, 1, &l1_residual) );
    
    CHECK_CUDA( cudaFree(d_TEMP) );
    
    return frobenNorm_DIFF / frobenNorm_A;
    
}


double svd_l1_err(cublasHandle_t &cublasH, const double *A,
                  const double *U, const double *S, const double *Vt,
                  const uint64_t m, const uint64_t n, const uint64_t k){
    
    const uint64_t ldA = roundup_to_32X( m );
    const uint64_t ldS = roundup_to_32X( k );
    const uint64_t ldvt= roundup_to_32X( n );
    
    double *d_TEMP, *d_DIFF;
    
    CHECK_CUDA( cudaMalloc( (void**)&d_TEMP, ldA * n * sizeof(double)) );
    CHECK_CUDA( cudaMalloc( (void**)&d_DIFF, ldA * n * sizeof(double)) );
    CHECK_CUDA( cudaMemsetAsync(d_TEMP, 0, ldA * n * sizeof(double)) );
    
    double l1Norm_A = 0.0;

    // copy A to d_TEMP to remove padding
    const double f_one = 1.0, f_zero = 0.0;
    CHECK_CUBLAS( cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                              m, n, // output dimension
                              &f_one,  A,   ldA,
                              &f_zero, NULL,ldA,
                              d_TEMP, m ) );
    // calculate l1 norm of A
    // the l1 Norm of d_TEMP is calculated by treating matrix as a vector
    CHECK_CUBLAS( cublasDasum(cublasH, m * n, d_TEMP, 1, &l1Norm_A) );
    
    //cout << "l1 norm of A = " << l1Norm_A << endl;
    
    // copy A to d_DIFF
    CHECK_CUDA( cudaMemcpy(d_DIFF, A, ldA * n * sizeof(double), cudaMemcpyDeviceToDevice) );
    // print_device_matrix(d_A, m, n, m, "A_without_padding");
    
    // S is transformed into a diagonal matrix
    // d_TEMP = S * VT,  SVt[k, n] = S[k, k] * Vt[k, n]
    CHECK_CUBLAS( cublasDdgmm( cublasH, CUBLAS_SIDE_LEFT,
                               k, n,
                               Vt,  ldvt,
                               S,   1,  //careful S is vector
                               d_TEMP, ldS) );
    
    // print_device_matrix(SVt, k, n, ldS, "S*Vt");
    
    // DIFF := -U * d_TEMP + A, DIFF[m, n] = -1 * U[m, k] * d_TEMP[k, n] + DIFF[m, n]
    const double alpha = -1.0, beta = 1.0;
    CHECK_CUBLAS( cublasDgemm_v2( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  U,      ldA,
                                  d_TEMP, ldS,
                                  &beta,
                                  d_DIFF, ldA) );
    
    // copy DIFF to d_TEMP inside device to remove padding
    // use matrix addition to realize copy function
    // there maybe better way to do padded copy (20161202, CUDA 8.0)
    CHECK_CUBLAS( cublasDgeam( cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n,     // output dimension
                               &f_one,  d_DIFF, ldA,
                               &f_zero, NULL,   ldA,
                                       d_TEMP, m ) );
    
    
    // print_device_matrix(DIFF, m, n, m, "DIFF");
    
    double l1Norm_DIFF = 0.0;
    // cublasDasum calculate the l1 norm of a vector
    CHECK_CUBLAS( cublasDasum(cublasH, m * n, d_TEMP, 1, &l1Norm_DIFF) );
    //cout << "l1 norm of A - U*S*VT = " << l1Norm_DIFF << endl;
    
    CHECK_CUDA( cudaFree(d_DIFF) );
    CHECK_CUDA( cudaFree(d_TEMP) );
    return  l1Norm_DIFF / l1Norm_A;
    
}

void orthogonalization(cusolverDnHandle_t &cusolverH,
                       double *d_A, const uint64_t m, const uint64_t n){
    
    const int ldA = roundup_to_32X(m);
    double *d_tau = NULL, *d_work = NULL;
    //double *d_R = NULL;
    int *devInfo =  NULL;
    //int lwork_geqrf = 0, lwork_orgqr = 0, lwork = 0, info_gpu = 0;
    int lwork;
    
    CHECK_CUDA( cudaMalloc((void**)&d_tau,   sizeof(double)*n) );
    CHECK_CUDA( cudaMalloc((void**)&devInfo, sizeof(int)) );
    //CHECK_CUDA( cudaMalloc((void**)&d_R,     sizeof(double)*n*n) );
    
    // step 1: query working space of geqrf and orgqr
//    CHECK_CUSOLVER( cusolverDnDgeqrf_bufferSize( cusolverH,
//                                                m, n,
//                                                d_A, ldA,
//                                                &lwork_geqrf) );
//    
//    CHECK_CUSOLVER( cusolverDnDorgqr_bufferSize( cusolverH,
//                                                m, n, n,
//                                                d_A, ldA,
//                                                d_tau,
//                                                &lwork_orgqr) );
    
    // choose the larger device work space
//    lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
//    cout << "m = " << m << ", n = " << n << endl;
//    cout << "lwork_geqrf = " << lwork_geqrf * sizeof(double) / pow(2, 30) << " GB" << endl;
//    cout << "lwork_orgqr = " << lwork_orgqr * sizeof(double) / pow(2, 30) << "GB." << endl;

    lwork = m * n;
    CHECK_CUDA( cudaMalloc((void**)&d_work, sizeof(double) * lwork) );
    
    //size_t freeMem, totalMem;
    //CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    //CHECK_CUDA( cudaMalloc((void**)&d_work, freeMem / 2) );
    
    // step 2: compute QR factorization
    CHECK_CUSOLVER( cusolverDnDgeqrf( cusolverH,
                                     m, n,
                                     d_A, ldA,
                                     d_tau, d_work, lwork,
                                     devInfo) );
    
    CHECK_CUDA( cudaGetLastError());
    // CHECK_CUDA( cudaDeviceSynchronize() );
    
    // copy info to host to check if QR is successful or not
   // CHECK_CUDA( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    
    //cout << "after geqrf: info_gpu = "<<  info_gpu << endl;;
   // assert(0 == info_gpu && "orthogonalization error ");
    
    // step 3: compute Q
    CHECK_CUSOLVER( cusolverDnDorgqr( cusolverH,
                                     m, n, n,
                                     d_A, ldA,
                                     d_tau, d_work, lwork,
                                     devInfo) );
    
    CHECK_CUDA( cudaGetLastError());
//    CHECK_CUDA( cudaDeviceSynchronize() );
//    
//    // step 4: check QR decomposition process
//    CHECK_CUDA( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
//    
//    //cout << "after orgqr: info_gpu = " <<  info_gpu << endl;
//    assert(0 == info_gpu &&"QR decomposition error ");
    
    // clean up
    CHECK_CUDA( cudaFree(d_tau) );
    CHECK_CUDA( cudaFree(devInfo) );
    CHECK_CUDA( cudaFree(d_work) );
    //CHECK_CUDA( cudaFree(d_R) );
    
}

// A will be overwritten inside svd inside SVD
void svd(cusolverDnHandle_t &cusolverH,
         double *d_U, double *d_S, double *d_VT,
         double *d_A, const uint64_t m, const uint64_t n){
    
    // cout << "m = " << m << ", n = " << n << endl;
    assert( m>=n && "cuslover can not SVD short-wide matrix with rows < columns, also be careful input matrix will be overwritten!." );
    
    const uint64_t lddA = roundup_to_32X( m );
    const uint64_t lddVT= roundup_to_32X( n );
    const uint64_t lddU = lddA;
    
    // step 1: query working space of SVD
    int lwork = 0;
    CHECK_CUSOLVER( cusolverDnDgesvd_bufferSize( cusolverH, m, n, &lwork ) );
    
    double *d_work = NULL;
    int *devInfo  = NULL;
    CHECK_CUDA( cudaMalloc((void**)&d_work , sizeof(double) * lwork) );
    CHECK_CUDA( cudaMalloc((void**)&devInfo, sizeof(int)) );
    
    // step 2: compute SVD
    //const signed char jobu =  'A';  // all m columns of U
    //const signed char jobvt = 'A';  // all n columns of VT
    const signed char jobu =  'S'; // the first min(m,n) columns of U (the left singular vectors) are returned in the array U
    const signed char jobvt = 'S'; // the first min(m,n) rows of V^T (the right singular vectors) are returned in the array VT
    
    CHECK_CUSOLVER( cusolverDnDgesvd( cusolverH, jobu, jobvt,
                                      m, n,
                                      d_A,  lddA,
                                      d_S,
                                      d_U,  lddU,
                                      d_VT, lddVT,
                                      d_work, lwork, NULL,
                                      devInfo) );
    
    int devInfo_h = 0;
    CHECK_CUDA( cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if (devInfo_h != 0) { std::cout    << "SVD fialed" << endl; }
    
    // clean up
    CHECK_CUDA( cudaFree(d_work) );
    CHECK_CUDA( cudaFree(devInfo) );
    
}

void transposeGPU(cublasHandle_t &cublasH,
                  double *AT, const double *A,
                  const uint64_t m, const uint64_t n){ // M, N is input dimension
    
    const double double_one = 1.0f, double_zero = 0.0f;
    
    const uint64_t ldA  = roundup_to_32X( m );
    const uint64_t ldAt = roundup_to_32X( n );
    
    CHECK_CUBLAS( cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_T,
                              n, m, // output dimension
                              &double_one,  A,   ldA,
                              &double_zero, NULL,ldA,
                              AT, ldAt ) );
    
}

void print_device_matrix(const double *d_A, const uint64_t m, const uint64_t n,
                         const uint64_t ldA, const char* name){
    
    /* debug test */
    double *host_A;
    CHECK_CUDA( cudaHostAlloc( (void**)&host_A, ldA * n * sizeof(double), cudaHostAllocPortable ) );
    
    // copy BT to host
    CHECK_CUBLAS( cublasGetMatrix( m, n, sizeof(*host_A), d_A, ldA, host_A, m) );
    
    cout << name << " =";
    print_column_major_matrix(host_A, m, n);
    
    CHECK_CUDA( cudaFreeHost(host_A) );
    
};


void print_column_major_matrix(const double *A, const uint64_t M, const uint64_t N){
    cout.precision(1);
    cout << "[" << endl;
    for (uint64_t i = 0; i < M; i++){
        for(uint64_t j = 0; j < N; j++){
            cout  << scientific <<A[j * M + i];
            if(j < N-1) { cout << ", ";      }
            else        { cout <<";" <<endl; }
        }
        
    }
    cout << "];"<< endl << endl;
}
