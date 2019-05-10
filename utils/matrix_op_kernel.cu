
// __global__ void transpose(Matrix M, Matrix P)
// {
 
//     // Get our global thread IDs
//     int idx = blockIdx.x*blockDim.x+threadIdx.x;
//     int idy = blockIdx.y*blockDim.y+threadIdx.y;

//     int temp = M.elements[*M.width+j];
//     __syncthreads();

//     // Make sure we do not go out of bounds
//     if(idx < M.width && idy < M.height){
//         P.elements[idx*P.width+idy] = temp;
//     }


// }
#define TILE_WIDTH 16

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

// // c = a + b * s
// void vmadd(const Vector& a, const Vector& b, double s, Vector& c)
// {
//   if (c.size != a.size or c.size != b.size) {
//     std::cerr << "[vmadd]: vector sizes don't match\n";
//     return;
//   }
 
//   for (int i = 0; i < c.size; i++)
//     c(i) = a(i) + s * b(i);
// }