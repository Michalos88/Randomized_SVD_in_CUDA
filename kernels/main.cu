#include "gpuErrorCheck.h"
#include "rsvd.h"
#include <fstream>
#include <chrono> // for timer

#define testError true

using namespace std;

// timer
uint64_t getCurrTime()
{
    return chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void rsvd_test(const uint64_t m, const uint64_t n, const uint64_t k, const double sparsity){
    
    // create cusolverDn/cublas handle
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    CHECK_CUSOLVER( cusolverDnCreate(&cusolverH) );
    CHECK_CUBLAS( cublasCreate(&cublasH) );
    
    const uint64_t p = k; // oversampling number
    const uint64_t l = k + p;
    const uint64_t q = 2; // power iteration factor
    //cout << "l = " << l << ", m = " << m << ", n = " << n << endl;
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
    
    //CHECK_CUDA(cudaDeviceReset());
    /*************************************** SVD by cublasXt ************************************************/
    tick = getCurrTime();

    cublasXtRsvd(host_U, host_S, host_VT, host_A, m, n, l, q, cusolverH, cublasH);
    tock = getCurrTime();
    double cublasXtTime = (tock - tick) / 1e6; // from ms to s
    
    memcpy(host_S1, host_S, l * sizeof(double) );
    
    //print_device_matrix(dev_S, 1, 5, 1, "Leading 5 singular values by OoC" );
    double cublasXtErr = 0.0;
    
    if(testError == true){
        cublasXtErr = svdFrobeniusDiff(host_A, host_U, host_S, host_VT, m, n, l);
    }
    //CHECK_CUDA(cudaDeviceReset());
    /******************************* Out-of-Core by me *********************************/
    tick = getCurrTime();
    // TODO find a better size
    const uint64_t batch = dataSize / (freeMem / 4);
    rsvdOoC(host_U, host_S, host_VT, host_A, m, n, l, q, batch, cusolverH, cublasH);
    //(host_U, host_S, host_VT, host_A, m, n, l, q, 4, cusolverH, cublasH);
    tock = getCurrTime();
    double myOoCTime = (tock - tick) / 1e6; // from ms to s

    double myOoCErr = 0;
    if(testError == true){
        myOoCErr = svdFrobeniusDiff(host_A, host_U, host_S, host_VT, m, n, l);
    }
    
    // error test
    // calculate L1 norm of singular value difference between cublasXt and my out-of-core
    //cout << "S1 = ";
    double singular_diff = 0.0;
    for (int i = 0; i < l; i++){
        //  cout << host_S1[i] << " ";
        singular_diff += fabs(host_S1[i] - host_S[i]);
    }
    
    CHECK_CUDA( cudaFreeHost(host_S1));
    CHECK_CUDA( cudaFreeHost(host_A) );
    CHECK_CUDA( cudaFreeHost(host_U) );
    CHECK_CUDA( cudaFreeHost(host_S) );
    CHECK_CUDA( cudaFreeHost(host_VT));
    
    // print to screen
    double dataGB = m * n * 8 / pow(2, 30);
    cout << setprecision(2) << dataGB << "\t"<< m << "\t" << n << "\t" << k << "\t"
    << scientific << setprecision(2)
    << InCoreTime <<"\t"<< cublasXtTime << "\t" << myOoCTime << "\t"
    << InCoreErr << "\t" << cublasXtErr << "\t" << myOoCErr << "\t" <<  singular_diff << endl;
    
     //save to file
        fstream fs;
        fs.open("RSVD_double.csv", fstream::out | fstream::app);
    
        fs << dataGB << ","
        << m << "," << n << "," << k << ","
        << InCoreTime << "," << cublasXtTime  << "," << myOoCTime << ","
        << InCoreErr << "," << cublasXtErr << "," << myOoCErr << ","
        << singular_diff << endl;
    
        fs.close();
    
    //CHECK_CUDA( cudaFreeHost(host_S1) );
    //CHECK_CUDA( cudaFreeHost(host_S2) );
    
    CHECK_CUBLAS( cublasDestroy(cublasH) );
    CHECK_CUSOLVER( cusolverDnDestroy(cusolverH) );
    
}

void rQR_test(void){
    
    const unsigned int m = 128, n = 128;
    const unsigned int ldA = m;
    const float float_one = 1.0, float_zero = 0, float_minus_one = -1.0;
    
    double *h_A = (double *)malloc( ldA * n * sizeof(double) );
    
    // device memory
    double *d_A;
    CHECK_CUDA( cudaMalloc((void **)&d_A, ldA * n * sizeof(double)));
    CHECK_CUDA( cudaMemset(d_A, 0, ldA * n * sizeof(double)) );
    
    //print_device_matrix(d_A, m, n, ldA, "d_A");
    double *dT;
    float *fT;
    
    cout << "sizeof(dT) = " << sizeof(dT) << endl;
    cout << "sizeof(fT) = " << sizeof(fT) << endl;
    cout << "sizeof(double) = " << sizeof(double) << endl;
    cout << "sizeof(float)  = " << sizeof(float) << endl;
    
    CHECK_CUBLAS( cublasSetMatrix(m, n, sizeof(double), h_A, m, d_A, m) );

    CHECK_CUDA( cudaFree(d_A) );
    
    free(h_A);
    
}

int main(int argc, char **argv) {
    
    assert((argc >= 3) && "Please supply M, N, K as the argument");
    
    uint64_t m = atoll(argv[1]);
    uint64_t n = atoll(argv[2]);
    uint64_t k = atoll(argv[3]);
    double    s = stof(argv[4]);
    
    //cudaDeviceProp deviceProp;
    
    //CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    
    //size_t gpuGlobalMem = deviceProp.totalGlobalMem;
    
    //fprintf(stderr, "GPU global memory = %zu Bytes\n", gpuGlobalMem);
    
    
    
    
    //double GB = double(1024 * 1024 * 1024);
    // fprintf(stderr, "Free = %0.1f GB, Total = %0.1f GB\n", double(freeMem) / GB, double(totalMem) / GB);
    
    
    rsvd_test(m, n, k, s);
    //rQR_test();
    
    CHECK_CUDA(cudaDeviceReset());
    
    return 0;
}