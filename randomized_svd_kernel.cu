#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrix.h"
#define TILE_WIDTH 16

////////////////////////////////////////////////////////////////////////////////
//  (1) generation of random matrix 𝛺;
//  (2) matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;
//  (3) QR decomposition on 𝑌;
//  (4) matrix-matrix multiplication of 𝑄𝑇𝐴; and
//  (5) deterministic SVD decomposition on 𝐵.
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void find_randomized_range(Matrix M, int rank)
{
 
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Generate Random Matrix alpha or copy from 
    
    // matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;

 
    //  for (int m = 0; m < (int)ceil((float)M.width / blockDim.x); ++m) {
    //      // Collaborative loading of Md and Nd tiles into shared memory
    //      if((m*TILE_WIDTH + tx) < M.width && row < P.height){
    //          Mds[ty][tx] = M.elements[row*M.width + (m*TILE_WIDTH + tx)];
    //      }else
    //      {
    //          Mds[ty][tx] = 0;
    //      }
    //      if((m*TILE_WIDTH + ty) < N.height && col < P.width){
    //          Nds[ty][tx] = N.elements[(m*TILE_WIDTH + ty)* N.width + col];
    //      }else
    //      {
    //          Nds[ty][tx] = 0;
    //      }

    //      __syncthreads();

    //      for(int k = 0; k <TILE_WIDTH; ++k){
    //          Pvalue += Mds[ty][k] * Nds[k][tx];
    //      }
    //      __syncthreads();

    //  }
    //  if(col<P.width && row<P.height){
    //      P.elements[row*P.width + col] = Pvalue;
    //  }
}
 
__global__ void qr_decomposition(Matrix M)
{
 
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Generate Random Matrix alpha or copy from 
    
    // matrix-matrix multiplication of 𝐴𝛺 to produce 𝑌;
    
 
    //  for (int m = 0; m < (int)ceil((float)M.width / blockDim.x); ++m) {
    //      // Collaborative loading of Md and Nd tiles into shared memory
    //      if((m*TILE_WIDTH + tx) < M.width && row < P.height){
    //          Mds[ty][tx] = M.elements[row*M.width + (m*TILE_WIDTH + tx)];
    //      }else
    //      {
    //          Mds[ty][tx] = 0;
    //      }
    //      if((m*TILE_WIDTH + ty) < N.height && col < P.width){
    //          Nds[ty][tx] = N.elements[(m*TILE_WIDTH + ty)* N.width + col];
    //      }else
    //      {
    //          Nds[ty][tx] = 0;
    //      }

    //      __syncthreads();

    //      for(int k = 0; k <TILE_WIDTH; ++k){
    //          Pvalue += Mds[ty][k] * Nds[k][tx];
    //      }
    //      __syncthreads();

    //  }
    //  if(col<P.width && row<P.height){
    //      P.elements[row*P.width + col] = Pvalue;
    //  }
}


__global__ void transpose(Matrix M, Matrix P)
{
 
    // Get our global thread IDs
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    int temp = M.elements[*M.width+j];
    __syncthreads();

    // Make sure we do not go out of bounds
    if(idx < M.width && idy < M.height){
        P.elements[idx*P.width+idy] = temp
    }


}

__global__ void MatrixMulKernel_Shared(Matrix M, Matrix N, Matrix P)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;

    for (int m = 0; m < (int)ceil((float)M.width / blockDim.x); ++m) {
        // Collaborative loading of Md and Nd tiles into shared memory
        if((m*TILE_WIDTH + tx) < M.width && row < P.height){
            Mds[ty][tx] = M.elements[row*M.width + (m*TILE_WIDTH + tx)];
        }else
		{
			Mds[ty][tx] = 0;
		}
        if((m*TILE_WIDTH + ty) < N.height && col < P.width){
            Nds[ty][tx] = N.elements[(m*TILE_WIDTH + ty)* N.width + col];
        }else
		{
			Nds[ty][tx] = 0;
		}

        __syncthreads();

        for(int k = 0; k <TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    if(col<P.width && row<P.height){
        P.elements[row*P.width + col] = Pvalue;
    }
}

// c = a + b * s
__global__ void vmadd(double *a, double *b, double *c, double s, double n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n){
        c[id] = a[id] + s * b[id];
    }
        
}

// compute minor // 2D Block or 1D with for loop
__global__ void compute_minor(Matrix M, Matrix P,int d)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    if(idx=>0 && idx < d){
        P.elements[idx*P.width+idx] = 1;
    }
    else if(idx=> d && idx < M.width){
        if (idy=> d && idy < M.height))
            P[idy*P.width+idx] = M[idy*P.width+idx];
    }

}

// P = I - 2*v*v^T // P is n x n
__global__ compute_householder_factor(float *v, Matrix P)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idy = blockIdx.y*blockDim.y+threadIdx.y;

    // Make sure we do not go out of bounds    
    if(idx < P.width && idy < P.height){
        P.elements[idy*P.width+idx] = -2 *  v[idx] * v[idy];
    }

    __syncthreads();
    if(idx < P.width){
        P.elements[idx*P.width+idx] += 1;
    }
}

// Has to be 1D block
__global__ void householder(Matrix M, Matrix R, Matrix Q)
{

    int m = M.width;
    int n = M.height;

    // // array of factor Q1, Q2, ... Qm
    // std::vector<Matrix> qv(m);

    // // temp array
    // Matrix z(mat);
    // Matrix z1;
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // for (int k = 0; k < n && k < m - 1; k++) {
    if (id < n && id <m-1){

        Vector e(m), x(m);
        double a;

        // compute minor
        z1.compute_minor(z, id;

        // extract k-th column into x
        z1.extract_column(x, id);

        a = x.norm();
        if (M.elements[idx*M.width+idx] > 0) a = -a;

        for (int i = 0; i < e.size; i++)
            e(i) = (i == k) ? 1 : 0;

        // e = x + a*e
        vmadd(x, e, a, e);

        // e = e / ||e||
        e.rescale_unit();

        // qv[k] = I - 2 *e*e^T
        compute_householder_factor(qv[k], e);

        // z = qv[k] * z1
        z.mult(qv[k], z1);

    }

    Q = qv[0];

    // after this loop, we will obtain Q (up to a transpose operation)
    for (int i = 1; i < n && i < m - 1; i++) {

    z1.mult(qv[i], Q);
    Q = z1;

    }

    R.mult(Q, mat);
    Q.transpose();

}



#endif // #ifndef _MATRIXMUL_KERNEL_H_
 