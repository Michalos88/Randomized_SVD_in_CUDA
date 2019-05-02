#include "randomized_svd.h" 
// includes, system
#include <string>

// includes, project
#include "matrix.h"
#include "nocutil.h"
// #include "matrixmul_gold.cpp"

// includes, kernels
#include "randomized_svd_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

// extern "C"
// void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

// void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M_host;
	int error_read = 0,
	
	srand(5672);
	// TODO: CASE2: Randomized size of a matrix
	if(argc != 4) {
		// Allocate and initialize the matrices
		// float M_height = rand() % 1024;
        // float M_width = rand() % 1024;
        float M_height = 1024;
		float M_width = 1024;

		// while(M_height*N_width>64000){
		// 	M_height = rand() % 1024;
		// 	M_width = rand() % 1024;
		// }
		
		M_host  = AllocateMatrix(M_height, M_width, 1);

	}else{
		// Allocate and read in matrices from disk
		int* params = NULL; //(int*)malloc(3 * sizeof(int));
	    unsigned int data_read = 3;
	    nocutReadFilei(argv[1], &params, &data_read, true);
		if(data_read != 3){
			printf("Error reading parameter file\n");
			return 1;
		}

		M_host  = AllocateMatrix(params[0], params[1], 0);
		error_read = ReadFile(&M_host, argv[2]);
		if(error_read){
			printf("Error reading input files %d\n", error_read);
			return 1;
		}
    }
    

	// FAST SVD on Device
	printf("CPU Computation Started\n");
	
	printf("GPU Computation Completed\n");

    // compute the matrix multiplication on the CPU for comparison
    // Matrix reference = AllocateMatrix(P.height, P.width, 0);
    // computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);

        

    int rank = 10;
	printf("CPU Computation Started\n");
	RandomizedSvd rsvd(M_host, rank);
	// printf("%f", rsvd.singularValues()[0]);
    // in this case check if the result is equivalent to the expected soluion
    // bool res = nocutComparefe(reference.elements, P.elements, 
	// 								P.height*P.width, 0.001f);
    // printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    printf("CPU Computation Completed\n");
    if(argc == 4)
    {
		WriteFile(M_host, argv[3]);
	}
	else if(argc == 2)
	{
	    WriteFile(M_host, argv[1]);
	}   

	// Free matrices
    FreeMatrix(&M_host);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
// void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
// {
//     // Load M and N to the device
//     Matrix Md = AllocateDeviceMatrix(M);
//     CopyToDeviceMatrix(Md, M);
//     Matrix Nd = AllocateDeviceMatrix(N);
//     CopyToDeviceMatrix(Nd, N);

//     // Allocate P on the device
//     Matrix Pd = AllocateDeviceMatrix(P);
// 	CopyToDeviceMatrix(Pd, P); // Clear memory
	
// 	int tileWidth = 16;

// 	// Setup the execution configuration
// 	dim3 dimBlock(tileWidth,tileWidth);
//     dim3 dimGrid((int)ceil((float)P.width / dimBlock.x), (int)ceil((float)P.height / dimBlock.y));

//     // Execute the kernel
//     MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);  
//     // Launch the device computation threads!

//     // Read P from the device
//     CopyFromDeviceMatrix(P, Pd); 

//     // Free device matrices
//     FreeDeviceMatrix(&Md);
//     FreeDeviceMatrix(&Nd);
//     FreeDeviceMatrix(&Pd);
// }

// Allocate a device matrix of same size as M.
// Matrix AllocateDeviceMatrix(const Matrix M)
// {
//     Matrix Mdevice = M;
//     int size = M.width * M.height * sizeof(float);
//     cudaMalloc((void**)&Mdevice.elements, size);
//     return Mdevice;
// }

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    
    // don't allocate memory on option 2
    if(init == 2)
		return M;
	
	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{	 
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);;
	}
    return M;
}	

// // Copy a host matrix to a device matrix.
// void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
// {
//     int size = Mhost.width * Mhost.height * sizeof(float);
//     Mdevice.height = Mhost.height;
//     Mdevice.width = Mhost.width;
//     cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
// 					cudaMemcpyHostToDevice);
// }

// // Copy a device matrix to a host matrix.
// void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
// {
//     int size = Mdevice.width * Mdevice.height * sizeof(float);
//     cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
// 					cudaMemcpyDeviceToHost);
// }

// // Free a device matrix.
// void FreeDeviceMatrix(Matrix* M)
// {
//     cudaFree(M->elements);
//     M->elements = NULL;
// }

// // Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = M->height*M->width;
	nocutReadFilef(file_name, &(M->elements), &data_read, true);
	return (data_read != (M->height * M->width));
}

// Write a floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    nocutWriteFilef(file_name, M.elements, M.width*M.height,
                       0.0001f);
}



// int RSVD_host(int argc, char* argv[]) {
  
//     // Randomized SVD
//     int rank = 10;
//     cout << "Randomized SVD with rank " << rank << ": ";
//     start = steady_clock::now();
  
//     RandomizedSvd rsvd(M, rank);
  
//     now = steady_clock::now();
//     elapsed = duration_cast<milliseconds>(now - start).count();
//     std::cout << elapsed << " ms" << std::endl;
  
//     cout << "Reconstruction error for full SVD (zero): " <<
//       diff_spectral_norm(M, full_svd.matrixU(), full_svd.singularValues(), full_svd.matrixV()) << endl;
//     cout << "Reconstruction error for rand SVD: " <<
//       diff_spectral_norm(M, rsvd.matrixU(), rsvd.singularValues(), rsvd.matrixV()) << endl;
  
//     return 0;
//   }