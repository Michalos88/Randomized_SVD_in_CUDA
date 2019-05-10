#include "gpuErrorCheck.h"
#include "rsvd.h"
#include <cublasXt.h>

using namespace std;

void genNormalRand(double *h_rand, const uint64_t m, uint64_t n){
    
    curandGenerator_t randGen;
    CHECK_CURAND( curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT) );
    CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)) );
    
    const uint64_t batchSize = pow(2, 30) / sizeof(double);// fixed 1GB data every batch
    const uint64_t lda = roundup_to_32X( m );
    
    double *d_rand;
    if( m * n < batchSize){
        
        // allocate device
        CHECK_CUDA( cudaMalloc((void**)&d_rand, lda * n * sizeof(double)) );
        // generate double normal distribution with mean = 0.0, stddev = 1.0
        CHECK_CURAND( curandGenerateNormalDouble(randGen, d_rand, lda * n, 0.0f, 1.0f) );
        CHECK_CUDA( cudaMemcpy(h_rand, d_rand, m * n * sizeof(double), cudaMemcpyDeviceToHost) );
        
    }else{
        
        CHECK_CUDA( cudaMalloc((void**)&d_rand, batchSize * sizeof(double)) );
        const uint64_t batch = (m * n) / batchSize;
        for(uint64_t i = 0; i < batch; i++){
            CHECK_CURAND( curandGenerateNormalDouble(randGen, d_rand, batchSize, 0.0f, 1.0f) );
            CHECK_CUDA( cudaMemcpy(h_rand + i * batchSize, d_rand,
                                   batchSize * sizeof(double), cudaMemcpyDeviceToHost) );
        }
        //last batch
        const uint64_t lastbatch = m * n - batch * batchSize;
        const uint64_t roundedlastbatch = roundup_to_32X( lastbatch );
        
        if(lastbatch != 0){
            CHECK_CURAND( curandGenerateNormalDouble(randGen, d_rand, roundedlastbatch, 0.0f, 1.0f) );
            CHECK_CUDA( cudaMemcpy(h_rand + batch * batchSize, d_rand,
                                   lastbatch * sizeof(double), cudaMemcpyDeviceToHost) );
        }
    }
    
    CHECK_CURAND( curandDestroyGenerator(randGen) );
    CHECK_CUDA( cudaFree(d_rand) );
    
}

void genUniformRand(double *h_rand, const uint64_t m, uint64_t n){
    
    curandGenerator_t randGen;
    CHECK_CURAND( curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT) );
    CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)) );
    
    const uint64_t batchSize = pow(2, 30) / sizeof(double);// fixed 1GB data
    const uint64_t lda = roundup_to_32X( m );
    double *d_rand;
    if( m * n < batchSize){
        
        // allocate device
        CHECK_CUDA( cudaMalloc((void**)&d_rand, lda * n * sizeof(double)) );
        // generate double Uniform distribution
        CHECK_CURAND( curandGenerateUniformDouble(randGen, d_rand, lda * n) );
        CHECK_CUDA( cudaMemcpy(h_rand, d_rand, m * n * sizeof(double), cudaMemcpyDeviceToHost) );
        
    }else{
        CHECK_CUDA( cudaMalloc((void**)&d_rand, batchSize * sizeof(double)) );
        const uint64_t batch = (m * n) / batchSize;
        for(uint64_t i = 0; i < batch; i++){
            CHECK_CURAND( curandGenerateUniformDouble(randGen, d_rand, batchSize) );
            CHECK_CUDA( cudaMemcpy(h_rand + i * batchSize, d_rand,
                                   batchSize * sizeof(double), cudaMemcpyDeviceToHost) );
        }
        //last batch
        const uint64_t lastbatch = m * n - batch * batchSize;
        const uint64_t roundedlastbatch = roundup_to_32X( lastbatch );
        
        if(lastbatch != 0){
            CHECK_CURAND( curandGenerateUniformDouble(randGen, d_rand, roundedlastbatch) );
            CHECK_CUDA( cudaMemcpy(h_rand + batch * batchSize, d_rand,
                                   lastbatch * sizeof(double), cudaMemcpyDeviceToHost) );
        }
    }
    
    CHECK_CURAND( curandDestroyGenerator(randGen) );
    CHECK_CUDA( cudaFree(d_rand) );
    
}

// This is not performance tuned
void genLowRankMatrix(double *h_A,
                      const uint64_t m, const uint64_t n, const uint64_t k){
    
    curandGenerator_t randGen;
    CHECK_CURAND( curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT) );
    
    // seed
    CHECK_CURAND( curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)) );
    
    
    // allocate host as Pinned
    double *h_rand1, *h_rand2;
//    CHECK_CUDA( cudaHostAlloc( (void**)&h_rand1, m * k * sizeof(double), cudaHostAllocPortable ) );
//    CHECK_CUDA( cudaHostAlloc( (void**)&h_rand2, k * n * sizeof(double), cudaHostAllocPortable ) );
    CHECK_CUDA( cudaMallocHost((void**)&h_rand1, m * k * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&h_rand2, k * n * sizeof(double)) );
    
    // allocate device
    genNormalRand(h_rand1, m, k);
    genNormalRand(h_rand2, k, n);
    
    cublasXtHandle_t cublasXtH = NULL;
    cublasXtCreate(&cublasXtH);
    // setup device
    int devices[1] = { 0 };
    CHECK_CUBLAS( cublasXtDeviceSelect(cublasXtH, 1, devices) );
    // data = rand1 * rand2, data:[M, N],  rand1:[M, K], rand2:[K, N]
    const double double_one = 1.0, double_zero = 0.0;
    CHECK_CUBLAS( cublasXtDgemm(cublasXtH, CUBLAS_OP_N, CUBLAS_OP_N,
                                m, n, k,
                                &double_one,
                                h_rand1,   m,
                                h_rand2,   k,
                                &double_zero,
                                h_A, m) );
    
    CHECK_CUBLAS( cublasXtDestroy(cublasXtH) );
    
    CHECK_CUDA( cudaFreeHost(h_rand1) );
    CHECK_CUDA( cudaFreeHost(h_rand2) );
    
}

double frobeniusNorm(const double *h_A, const uint64_t m, const uint64_t n){
    
    const uint64_t batchSize = pow(2, 30) / sizeof(double); // fixed 1GB data
    
    double frobenNorm_A = 0.0;
    double *d_data = NULL;
    cublasHandle_t cublasH = NULL;
    CHECK_CUBLAS( cublasCreate(&cublasH) );
    
    if( m * n < batchSize){ // process in one batch

        // allocate device
        CHECK_CUDA( cudaMalloc((void**)&d_data, m * n * sizeof(double)) );

        CHECK_CUDA( cudaMemcpy(d_data, h_A, m * n * sizeof(double), cudaMemcpyHostToDevice) );

        CHECK_CUBLAS( cublasDnrm2(cublasH,  m * n, d_data, 1, &frobenNorm_A) );

    }else{  // divide and conquer
        double Norm_temp = 0.0;
        CHECK_CUDA( cudaMalloc((void**)&d_data, batchSize * sizeof(double)) );
        const uint64_t batch = (m * n) / batchSize;
        for(uint64_t i = 0; i < batch; i++){
            
            CHECK_CUDA( cudaMemcpy(d_data, h_A + i * batchSize,
                                   batchSize * sizeof(double), cudaMemcpyHostToDevice) );
            
            CHECK_CUBLAS( cublasDnrm2(cublasH,  batchSize, d_data, 1, &Norm_temp) );
            
            frobenNorm_A += Norm_temp * Norm_temp;
        
        }
        //last batch
        const uint64_t lastbatch = m * n - batch * batchSize;
        if(lastbatch != 0){
            CHECK_CUDA( cudaMemcpy(d_data, h_A + batch * batchSize,
                                   lastbatch * sizeof(double), cudaMemcpyHostToDevice) );
            CHECK_CUBLAS( cublasDnrm2(cublasH, lastbatch, d_data, 1, &Norm_temp) );
            frobenNorm_A += Norm_temp * Norm_temp;
        }
        frobenNorm_A = sqrt(frobenNorm_A);
    }
    
    CHECK_CUDA( cudaFree(d_data) );
    CHECK_CUBLAS( cublasDestroy(cublasH) );
    
    //printf("frobenNorm = %0.3f\n", frobenNorm_A);
    return frobenNorm_A;
}

double svdFrobeniusDiff(const double *A,
                        const double *U, const double *S, const double *Vt,
                        const uint64_t m, const uint64_t n, const uint64_t l){

    //cout << "U = " << frobeniusNorm(U, m, l) << endl;
    //cout << "Vt = " << frobeniusNorm(Vt, l, n) << endl;
    cublasXtHandle_t cublasXtH = NULL;
    cublasXtCreate(&cublasXtH);
    // setup device
    int devices[1] = { 0 };
    CHECK_CUBLAS( cublasXtDeviceSelect(cublasXtH, 1, devices) );
    
    double *h_Sv, *h_TEMP;
    
    CHECK_CUDA( cudaMallocHost((void**)&h_Sv, l * l * sizeof(double)) );
    CHECK_CUDA( cudaMallocHost((void**)&h_TEMP,l * n * sizeof(double)) );
    
    memset(h_Sv,   0, l * l * sizeof(double));
    memset(h_TEMP, 0, l * n * sizeof(double));
    
    // TODO: change to matrix-vector multiplcation
    for(uint64_t i = 0; i < l; i++){
        h_Sv[i + l * i] = S[i];
    }
    
    const double double_one = 1.0, double_zero = 0.0, double_minus_one = -1.0;
    // h_TEMP = h_Sv * VT, h_TEMP[l, n] = S[l, l] * Vt[l, n]
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_N, CUBLAS_OP_N,
                                 l, n, l,
                                 &double_one,
                                 h_Sv, l,
                                 Vt,   l,
                                 &double_zero,
                                 h_TEMP, l));
    
    //cout << "S = " << frobeniusNorm(S, l, 1) << endl;
    //cout << "h_TEMP = " << frobeniusNorm(h_TEMP, l, n) << endl;
    
    CHECK_CUDA( cudaFreeHost(h_Sv) );
    
    double *h_DIFF;
    CHECK_CUDA( cudaMallocHost((void**)&h_DIFF, m * n * sizeof(double)) );
    //memset(h_DIFF, 0, m * n * sizeof(double));
    memcpy(h_DIFF, A, m * n * sizeof(double));
    // DIFF := -U * h_TEMP + A, DIFF[m, n] = -1 * U[m, l] * h_TEMP[l, n] + A[m, n]
    CHECK_CUBLAS( cublasXtDgemm( cublasXtH, CUBLAS_OP_N, CUBLAS_OP_N,
                                m, n, l,
                                &double_minus_one,
                                U,      m,
                                h_TEMP, l,
                                &double_one,
                                h_DIFF, m) );
    
    CHECK_CUDA( cudaFreeHost(h_TEMP) );
    CHECK_CUBLAS( cublasXtDestroy(cublasXtH) );
    //cout << "Diff = " << frobeniusNorm(h_DIFF, m, n) << endl;
    //cout << "A = " << frobeniusNorm(A, m, n) << endl;
    
    double temp = frobeniusNorm(h_DIFF, m, n) / frobeniusNorm(A, m, n);
    
    CHECK_CUDA( cudaFreeHost(h_DIFF) );

    return temp;
    
}
