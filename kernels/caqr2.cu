#include "assert.h"
#include "../utils/gpuErrorCheck.h"
#include "../utils/rsvd.h"
#include <vector>
#include <iostream>
#include <cuda.h>

/* Tuned settings for float and double */
/*
#define FLOAT
#define real_t float
#define BLK_WIDTH 16
#define BLK_HEIGHT 128
#define NUM_THREADS 128
#define BLK_SIZE (BLK_WIDTH * BLK_HEIGHT)
#define BLK_ROWS (NUM_THREADS / BLK_WIDTH)
#define THREAD_STORAGE BLK_WIDTH
#define TID_L_MASK 0xF
#define TID_U_SHIFT 4
#define ZERO_INIT {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
*/

#define DOUBLE
#define real_t double
#define BLK_WIDTH 8
#define BLK_HEIGHT 64
#define NUM_THREADS 64
#define BLK_SIZE (BLK_WIDTH * BLK_HEIGHT)
#define BLK_ROWS (NUM_THREADS / BLK_WIDTH)
#define THREAD_STORAGE BLK_WIDTH
#define TID_L_MASK 0x7
#define TID_U_SHIFT 3
#define ZERO_INIT {0,0,0,0,0,0,0,0}

/******************************************* class ****************************************************/
class level_t {
public:
    int num_blocks;
    int aggregate_blocks;
    int block_offset;
    int lev;
    level_t();
    level_t(int nb, int ab, int bo, int l);
};

using namespace std;

/* This defines the main matrix data structure and functions to
 access and modify it */
class QR_matrix {
    
public:
    /* Input matrix properties */
    int m;
    int n;
    int lda;
    int ld_panel_size;
    int ldq;
    int ldq_panel;
    int blks_tall_total;
    int blks_wide_total;
    int internal_matrix_size;
    int total_blocks;
    real_t * mat_base;
    real_t * Q_base;
    
    /* Current information */
    real_t * mat_cur;
    real_t * Q;
    int m_current;
    int n_current;
    int blks_tall_cur;
    int blks_wide_cur;
    vector<level_t*> levels;
    
    /* Constructors*/
    QR_matrix();
    QR_matrix(real_t * h_A, const int m, const int n, const int lda);
    ~QR_matrix();
    
    /* Set the matrix in its internal form (transpose) and retrieve it back */
    void factor();
    void retrieveQ();
    void calculate_dimensions(const int m, const int n);
    
    void panelTranspose(const real_t * mat_in, const int m, const int n, const int lda);
    void panelTransInv(real_t * mat_out, const int m, const int n, const int lda);
    
    void retrieveR(real_t * mat_out, const int m, const int n, const int lda);
    real_t * blockQ(const int l);
    /* Update the pointer to the next panel */
    void increment(bool levelChagneFlag);
    void decrement(bool levelChagneFlag);
    int set_levels();
    
};


/*************************************** CUDA kernels *****************************************************/
__device__ real_t reduce(real_t * u_sh, real_t ub[], real_t col[], real_t * av, int tid_u, int tid_l, int tid)
{
#pragma unroll
    for(int i = 0 ; i < THREAD_STORAGE ; i++)
        ub[i] = u_sh[tid_u + i*BLK_ROWS];
    
    real_t val = (real_t) 0;
#pragma unroll
    for(int i = 0 ; i < THREAD_STORAGE ; i++)
        val += ub[i] * col[i];
    
    if(tid >= (NUM_THREADS/2)) av[tid] = val;
    __syncthreads();
    if(tid < (NUM_THREADS/2)) av[tid] = av[tid + (NUM_THREADS/2)] + val;
    __syncthreads();
    val = 0;
#pragma unroll
    for(int i = 0 ; i < (BLK_ROWS/2) ; i++)
        val += av[tid_l + BLK_WIDTH*i];
    
    return val;
}

__device__ void update(real_t * u_sh, real_t ub[], real_t col[], real_t res, int tid_u)
{
    // Rank-1 update
    real_t fres = (real_t) 2 * res;
#pragma unroll
    for(int i = 0 ; i < THREAD_STORAGE ; i++)
        col[i] -= (real_t) ub[i] * fres;
}

__device__ void load_a(real_t * a, real_t col[], int tid)
{
#pragma unroll
    for(int i = 0 ; i < THREAD_STORAGE; i++)
        col[i] = a[tid + i*NUM_THREADS];
}

__device__ void load_a_triangles(real_t * a, real_t col[], int tid, int offset_blocks, real_t * A_max)
{
    // This is hardcoded for 128x16. Oops!
    real_t * a_orig = a;
#pragma unroll
    for(int i = 0 ; i < THREAD_STORAGE / (BLK_WIDTH/BLK_ROWS); i++) {
        if(a < A_max) {
            for(int ii = 0 ; ii < (BLK_WIDTH/BLK_ROWS) ; ii++)
                col[(BLK_WIDTH/BLK_ROWS)*i + ii ] = a[tid + ii*NUM_THREADS];
        }
        a += offset_blocks * BLK_SIZE;
    }
    a = a_orig;
}

__device__ void write_a(real_t * a, real_t col[], int tid)
{
#pragma unroll
    for(int i = 0 ; i < THREAD_STORAGE ; i++)
        a[tid + i*NUM_THREADS] = col[i];
}

__device__ void write_a_triangles(real_t * a, real_t col[], int tid, int offset_blocks, real_t * A_max)
{
    // This is hardcoded for 128x16. Oops!
#pragma unroll
    for(int i = 0 ; i < THREAD_STORAGE/ (BLK_WIDTH/BLK_ROWS); i++) {
        if(a < A_max) {
            for(int ii = 0 ; ii < (BLK_WIDTH/BLK_ROWS) ; ii++)
                a[tid + ii*NUM_THREADS] = col[(BLK_WIDTH/BLK_ROWS)*i + ii ];
        }
        a += offset_blocks * BLK_SIZE;
    }
}

__device__ void load_u(real_t * u_sh, real_t * b, int tid)
{
#pragma unroll
    for(int i = 0 ; i < BLK_WIDTH ; i++)
        u_sh[tid + i*BLK_HEIGHT] = b[tid + i*BLK_HEIGHT];
    __syncthreads();
}

__global__ void hh_update_dense_reverse(real_t * a, real_t * b, int lda_panel, int ldq)
{
    __shared__ real_t av[NUM_THREADS];
    __shared__ real_t u[BLK_HEIGHT * BLK_WIDTH];
    real_t col[THREAD_STORAGE];
    real_t ub[THREAD_STORAGE];
    real_t * u_sh;
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    
    // Pretend we are in panel-transpose form
    a += blockIdx.x * BLK_SIZE + blockIdx.y * lda_panel;
    
    // Pretend we are in column major form
    b += blockIdx.x * BLK_SIZE;
    
    // load in the block
    load_a(a, col, tid);
    
    // Load u
    u_sh = &u[0];
    load_u(u_sh, b, tid);
    u_sh = &u[(BLK_WIDTH-1) * BLK_HEIGHT];
    
    // For each Householder vector
    for(int j = 0 ; j < BLK_WIDTH; j++) {
        
        // Matrix-vector multiply
        real_t res = reduce(u_sh, ub, col, av, tid_u, tid_l, tid);
        
        // Rank-1 update
        update(u_sh, ub, col, res, tid_u);
        
        // Go to the next Householder vector
        u_sh -= BLK_HEIGHT;
    }
    
    // write out the block
    write_a(a, col, tid);
}

__device__ void compute_u(real_t * u_sh, real_t col[], real_t norms[], int tid, int tid_u, int tid_l, int j, int row, int m)
{
    __shared__ real_t mulby_sh;
    __syncthreads();
    
    if(j + row >= m) {
        u_sh[tid] = (real_t) 0;
    }
    else {
        if(tid_l == j) {
            real_t local = 0.0;
#pragma unroll
            for(int i = 0 ; i < THREAD_STORAGE ; i++) {
                u_sh[tid_u + i*BLK_ROWS] = col[i];
                if(tid_u + i*BLK_ROWS > j) local += col[i] * col[i];
            }
            norms[tid_u] = local;
        }
        __syncthreads();
        
        if(tid == j)
        {
            real_t nm2_nminus1 = (real_t)0.0;
#pragma unroll
            for(int i = 0 ; i < BLK_ROWS ; i++)
                nm2_nminus1 += norms[i];
            
            real_t top_element = u_sh[j];
#ifdef DOUBLE
            real_t nm = sqrt(nm2_nminus1 + top_element*top_element);
#endif
#ifdef FLOAT
            real_t nm = sqrtf(nm2_nminus1 + top_element*top_element);
#endif
            u_sh[j] = top_element = (top_element >= (real_t)0) ? top_element + nm : top_element - nm;
#ifdef DOUBLE
            real_t divby = sqrt(nm2_nminus1 + top_element*top_element);
#endif
#ifdef FLOAT
            real_t divby = sqrtf(nm2_nminus1 + top_element*top_element);
#endif
            mulby_sh = (divby != (real_t) 0) ? ((real_t) 1.0) / divby : (real_t)0;
        }
        if(tid < j) u_sh[tid] = (real_t) 0;
        __syncthreads();
        u_sh[tid] *= mulby_sh;
    }
    __syncthreads();
}

// factor a small matrix block, householder vector is saved on b
__global__ void hh_factor_dense(real_t * a, real_t * b, int m, int lda_panel, int ldq)
{
    __shared__ real_t av[NUM_THREADS];
    __shared__ real_t u[BLK_HEIGHT];
    __shared__ real_t norms[BLK_WIDTH];
    
    real_t col[THREAD_STORAGE];
    real_t ub[THREAD_STORAGE];
    real_t * u_sh = &u[0];
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    int row = blockIdx.x * BLK_HEIGHT;
    
    // Pretend we are in panel-transpose form
    a += blockIdx.x * BLK_SIZE + blockIdx.y * lda_panel;
    
    // Pretend we are in column major form
    b += blockIdx.x * BLK_SIZE;
    
    // load in the block
    load_a(a, col, tid);
    
    // For each column of a
    for(int j = 0 ; j < BLK_WIDTH ; j++) {
        
        // Form (transpose) the u vector
        compute_u(u_sh, col, norms, tid, tid_u, tid_l, j, row, m);
        
        // Matrix-vector multiply: res = v' * A(i:m, :);
        real_t res = reduce(u_sh, ub, col, av, tid_u, tid_l, tid);
        
        // Rank-1 update: A(j:m, :) -= 2 * v * res;
        update(u_sh, ub, col, res, tid_u);
        
        // Write out u
        b[tid] = u_sh[tid];
        
        // Go to the next Householder vector
        b += BLK_HEIGHT;
        
    }
    
}

//
__global__ void hh_update_dense(real_t * a, real_t * b, int lda_panel, int ldq, int max_y)
{
    __shared__ real_t av[NUM_THREADS];
    __shared__ real_t u[BLK_HEIGHT * BLK_WIDTH];
    real_t col[THREAD_STORAGE];
    real_t ub[THREAD_STORAGE];
    real_t * u_sh;
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    
    // Pretend we are in panel-transpose form
    a += blockIdx.x * BLK_SIZE + blockIdx.y * lda_panel;
    
    // Pretend we are in column major form
    b += blockIdx.x * BLK_SIZE;
    
    // Load u
    u_sh = &u[0];
    load_u(u_sh, b, tid);
    
    for(int p = blockIdx.y ; p < max_y ; p += gridDim.y)
    {
        
        // load in the block
        load_a(a, col, tid);
        
        // For each Householder vector
        for(int j = 0 ; j < BLK_WIDTH; j++) {
            
            // Matrix-vector multiply
            real_t res = reduce(u_sh, ub, col, av, tid_u, tid_l, tid);
            
            // Rank-1 update
            update(u_sh, ub, col, res, tid_u);
            
            // Go to the next Householder vector
            u_sh += BLK_HEIGHT;
        }
        
        // write out the block
        write_a(a, col, tid);
        u_sh = &u[0];
        a += gridDim.y * lda_panel;
    }
}

__global__ void hh_factor_triangle(real_t * a, real_t * b, int m, int lda_panel, int ldq, int offset_blocks, real_t * A_max)
{
    __shared__ real_t av[NUM_THREADS];
    __shared__ real_t u[BLK_HEIGHT];
    __shared__ real_t norms[BLK_WIDTH];
    real_t col[THREAD_STORAGE] = ZERO_INIT;
    real_t ub[THREAD_STORAGE];
    real_t * u_sh = &u[0];
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    int row = blockIdx.x * (BLK_HEIGHT / BLK_WIDTH) * offset_blocks * BLK_HEIGHT;
    
    // Pretend we are in panel-transpose form
    a += row * BLK_WIDTH + blockIdx.y * lda_panel;
    
    // Pretend we are in column major form
    b += blockIdx.x * BLK_SIZE;
    
    // load in the block
    load_a_triangles(a, col, tid, offset_blocks, A_max);
    
    // For each column of a
    for(int j = 0 ; j < BLK_WIDTH ; j++) {
        
        // Form (transpose) the u vector
        compute_u(u_sh, col, norms, tid, tid_u, tid_l, j, row, m);
        
        // Matrix-vector multiply
        real_t res = reduce(u_sh, ub, col, av, tid_u, tid_l, tid);
        
        // Rank-1 update
        update(u_sh, ub, col, res, tid_u);
        
        // Write out u
        b[tid] = u_sh[tid];
        
        // Go to the next Householder vector
        b += BLK_HEIGHT;
    }
}

__global__ void hh_update_triangle_reverse(real_t * a, real_t * b, int lda_panel, int ldq, int offset_blocks, real_t * A_max)
{
    __shared__ real_t av[NUM_THREADS];
    __shared__ real_t u[BLK_HEIGHT * BLK_WIDTH];
    real_t col[THREAD_STORAGE] = ZERO_INIT;
    real_t ub[THREAD_STORAGE];
    real_t * u_sh;
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    
    // Pretend we are in panel-transpose form
    a += blockIdx.x * BLK_ROWS * offset_blocks * BLK_SIZE + blockIdx.y * lda_panel;
    
    // Update A_max in case we are working on a different column
    A_max += blockIdx.y * lda_panel;
    
    // Pretend we are in column major form
    b += blockIdx.x * BLK_SIZE;
    
    // load in the block
    load_a_triangles(a, col, tid, offset_blocks, A_max);
    
    // Load u
    u_sh = &u[0];
    load_u(u_sh, b, tid);
    u_sh = &u[(BLK_WIDTH-1)*BLK_HEIGHT];
    
    // For each Householder vector
    for(int j = 0 ; j < BLK_WIDTH; j++) {
        
        // Matrix-vector multiply
        real_t res = reduce(u_sh, ub, col, av, tid_u, tid_l, tid);
        
        // Rank-1 update
        update(u_sh, ub, col, res, tid_u);
        
        // Go to the next Householder vector
        u_sh -= BLK_HEIGHT;
    }
    
    // write out the block
    write_a_triangles(a, col, tid, offset_blocks, A_max);
}

__global__ void hh_update_triangle(real_t * a, real_t * b, int lda_panel, int ldq, int offset_blocks, real_t * A_max)
{
    __shared__ real_t av[NUM_THREADS];
    __shared__ real_t u[BLK_HEIGHT * BLK_WIDTH];
    real_t col[THREAD_STORAGE] = ZERO_INIT;
    real_t ub[THREAD_STORAGE];
    real_t * u_sh;
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    
    // Pretend we are in panel-transpose form
    a += blockIdx.x * BLK_ROWS * offset_blocks * BLK_SIZE + blockIdx.y * lda_panel;
    
    // Update the max in case we are on a different panel
    A_max += blockIdx.y * lda_panel;
    
    // Pretend we are in column major form
    b += blockIdx.x * BLK_SIZE;
    
    // load in the block
    load_a_triangles(a, col, tid, offset_blocks, A_max);
    
    // Load u
    u_sh = &u[0];
    load_u(u_sh, b, tid);
    
    // For each Householder vector
    for(int j = 0 ; j < BLK_WIDTH; j++) {
        
        // Matrix-vector multiply
        real_t res = reduce(u_sh, ub, col, av, tid_u, tid_l, tid);
        
        // Rank-1 update
        update(u_sh, ub, col, res, tid_u);
        
        // Go to the next Householder vector
        u_sh += BLK_HEIGHT;
    }
    
    // write out the block
    write_a_triangles(a, col, tid, offset_blocks, A_max);
}

/* Get the address of a block (i,j) for level l in the Q matrix */
real_t * QR_matrix::blockQ(const int l) {
    assert(l <= levels.size());
    int agg_blocks = levels[l]->aggregate_blocks;
    
    // Get pointer to the next level
    int offset_blocks = agg_blocks * BLK_SIZE;
    return Q + offset_blocks;
}

/* Panel transpose of a block
 <<< dim3(blks_tall_total, blks_wide_total), BLK_HEIGHT >>> */
__global__ void blockTranspose(real_t * out, const real_t * in,
                               int ld_panel_size, int m, int n, int ld_col)
{
    // Shared memory
    __shared__ real_t sh[(BLK_HEIGHT + 1) * BLK_WIDTH];// why + 1? to avoid bank conflict?
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    
    // Offset the input vector address
    in += BLK_HEIGHT * blockIdx.x + BLK_WIDTH * ld_col * blockIdx.y;
    out+= BLK_SIZE * blockIdx.x + ld_panel_size * blockIdx.y;
    
    // If we are close to the border then this will be < BLK_WIDTH
    int n_it = n - BLK_WIDTH * blockIdx.y;
    
    // Load whole block into shared memory into column major
    if(tid + BLK_HEIGHT * blockIdx.x < m) {
#pragma unroll
        for(int i = 0 ; i < BLK_WIDTH; i++)
            sh[tid + i*BLK_HEIGHT+i] = (i < n_it) ? in[tid + i * ld_col] : (real_t) 0;
    } else {
#pragma unroll
        for(int i = 0 ; i < BLK_WIDTH ; i++)
            sh[tid + i*BLK_HEIGHT+i] = (real_t) 0;
    }
    
    __syncthreads();
    
    // Load block out of shared memory in transposed form
    int off = tid_l * (BLK_HEIGHT+1) + tid_u;
    
#pragma unroll
    for(int i = 0 ; i < BLK_WIDTH ; i++)
    { out[tid + i * BLK_HEIGHT] = sh[off + i * BLK_ROWS]; }
    
}

/* Panel transpose of the entire matrix (inverse) */
__global__ void trans_inv(real_t * out, const real_t * in, int ld_panel_size, int m, int n, int ld_col)
{
    __shared__ real_t sh[(BLK_HEIGHT+1) * BLK_WIDTH];
    int tid = threadIdx.x;
    int tid_l = tid & TID_L_MASK;
    int tid_u = tid >> TID_U_SHIFT;
    
    // Offset the output matrix
    in+= BLK_SIZE * blockIdx.x + ld_panel_size * blockIdx.y;
    out+= BLK_HEIGHT * blockIdx.x + BLK_WIDTH * ld_col * blockIdx.y;
    
    // In case we run off the end in the n direction
    int n_it = n - BLK_WIDTH * blockIdx.y;
    n_it = (n_it < BLK_WIDTH) ? n_it : BLK_WIDTH;
    
    // Load block into shared memory in column major
    int off = tid_l * (BLK_HEIGHT+1) + tid_u;
    
#pragma unroll
    for(int i = 0 ; i < BLK_WIDTH ; i++)
        sh[off + i * BLK_ROWS] = in[tid + i * BLK_HEIGHT];
    
    __syncthreads();
    
    // Wrtite back out
    if(tid + BLK_HEIGHT * blockIdx.x >= m) return;
    for(int i = 0 ; i < n_it; i++)
        out[tid + i * ld_col] = sh[tid + i*BLK_HEIGHT+i];
    
}

/* Set matrix to identity */
__global__ void set_ident(real_t * A, int ld_panel_size, int m, int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Offset the input vector
    A += (BLK_WIDTH * BLK_WIDTH + ld_panel_size) * bid;
    if(tid + BLK_WIDTH * bid >= n) return;
    
    // Set diagonal
    int index = tid + tid * BLK_WIDTH;
    A[index] = 1.0;
    
}

/*************************************** Host functions *******************************************/

int orth_CAQR_size(const int m, const int n){
    
    int blks_tall_cur = ((m + BLK_HEIGHT - 1) / BLK_HEIGHT);
    //int blks_wide_cur = (n + BLK_WIDTH - 1) / BLK_WIDTH;
    int num_blocks = blks_tall_cur;
    int tb = num_blocks;
    
    // Add the first block
    level_t *lev = new level_t(num_blocks, 0, 0, 0);
    vector<level_t*> levels;
    levels.push_back(lev);
    int block_offset = 1;
    int levnum = 1;
    
    while(num_blocks > 1) {
        num_blocks = (num_blocks + (BLK_HEIGHT/BLK_WIDTH) - 1) / (BLK_HEIGHT / BLK_WIDTH);
        lev = new level_t(num_blocks, tb, block_offset, levnum);
        levels.push_back(lev);
        block_offset *= (BLK_HEIGHT/BLK_WIDTH);
        levnum++;
        tb += num_blocks;
    }
    
    //int blks_tall_total = ((m + BLK_HEIGHT - 1) / BLK_HEIGHT) + 1; // + 1 is necessary! find out why!
    //int ld_panel_size = blks_tall_total * BLK_WIDTH * BLK_HEIGHT;
    int ldq = tb * BLK_HEIGHT;
    int blks_wide_total = (n + BLK_WIDTH - 1) / BLK_WIDTH;
    int ldq_panel = ldq * BLK_WIDTH;
    return (ldq_panel * blks_wide_total);
    
}

//matrix is transposed inside every matrix block,
void QR_matrix::panelTranspose(const real_t * mat_in, const int m, const int n, const int lda) {
    
    assert(lda >= m);
    //CHECK_CUDA( cudaMemset(mat_base, 0, internal_matrix_size * sizeof(real_t)));
    //calculate_dimensions(m, n);
    
    // grid is set to the block number
    // 1 threadblock is in charge of transpose 1 matrix block,
    
    blockTranspose <<< dim3(blks_tall_total, blks_wide_total), BLK_HEIGHT >>>
    (mat_base, mat_in, ld_panel_size, m, n, lda);
    
    CHECK_CUDA( cudaThreadSynchronize() );
    CHECK_CUDA( cudaGetLastError() );
    
    
}

void QR_matrix::calculate_dimensions(const int m, const int n) {
    this->m = m;
    this->n = n;
    m_current = m;
    n_current = n;
    blks_wide_total = (n + BLK_WIDTH - 1) / BLK_WIDTH;
    blks_tall_total = ((m + BLK_HEIGHT - 1) / BLK_HEIGHT) + 1; // + 1 is necessary! find out why!
    total_blocks = set_levels();
    ld_panel_size = blks_tall_total * BLK_WIDTH * BLK_HEIGHT; // the size of a panel, (which is transposed)
    ldq = total_blocks * BLK_HEIGHT;
    ldq_panel = ldq * BLK_WIDTH;
    internal_matrix_size = ld_panel_size * blks_wide_total;
}

QR_matrix::QR_matrix() {}

QR_matrix::QR_matrix(real_t * d_A, const int m, const int n, const int lda) {
    
    // Build the data structures
    calculate_dimensions(m, n);
    
    // Allocate the data matrix
    CHECK_CUDA( cudaMalloc((real_t**) &mat_base, internal_matrix_size * sizeof(real_t)));
    
    
    Q_base = d_A;
    //    CHECK_CUDA( cudaMalloc( (real_t**) &Q_base, ldq_panel * blks_wide_total * sizeof(real_t) ) );
    //    CHECK_CUDA( cudaMemset( Q_base, 0, ldq_panel * blks_wide_total * sizeof(real_t) ) );
    
    // Transpose
    panelTranspose(d_A, m, n, lda);
    
    // Allocate the Q matrix
    
    //    printf("A size = %d.\n", lda * n);
    //    printf("A' size = %d.\n", internal_matrix_size);
    //    printf("Q szie = %d.\n", ldq_panel * blks_wide_total);
    
    // Set "current" pointers
    mat_cur = mat_base;
    Q = Q_base;
    this->lda = lda;
    
}


#define SIMD_WIDTH 16
void QR_matrix::factor()
{
    
    for(int i = 0; i < blks_wide_total; i++) {
        
        // Factor two blocks on the left
        hh_factor_dense <<< blks_tall_cur, NUM_THREADS >>> (mat_cur, blockQ(0), m_current, ld_panel_size, ldq);
        
        // Update two blocks on the left and right
        int y_wid = 1;
        if(blks_tall_cur * blks_wide_cur < 2000) { y_wid =  blks_wide_cur;                      }
        else                                     { y_wid = (blks_wide_cur - 1) / SIMD_WIDTH + 1;}
        // **** Added from version 1.2 to get performance on large square **
        
        hh_update_dense <<< dim3(blks_tall_cur, y_wid), NUM_THREADS >>> (mat_cur, blockQ(0), ld_panel_size, ldq, blks_wide_cur);
        
        for (int lev = 1; lev < levels.size(); lev++) {
            
            level_t * cur_lev = levels[lev];
            // TODO
            hh_factor_triangle <<< dim3(cur_lev->num_blocks, 1), NUM_THREADS>>> (mat_cur, blockQ(lev), m_current, ld_panel_size, ldq, cur_lev->block_offset, mat_cur + m_current * BLK_WIDTH);
            // TODO
            hh_update_triangle <<< dim3(cur_lev->num_blocks, blks_wide_cur), NUM_THREADS>>> (mat_cur, blockQ(lev), ld_panel_size, ldq, cur_lev->block_offset, mat_cur + m_current * BLK_WIDTH);
        }
        
        // Next panel
        increment(false);
    }
    
    CHECK_CUDA( cudaThreadSynchronize() );
    CHECK_CUDA( cudaGetLastError() );
}

void QR_matrix::retrieveQ()
{
    
    /* set block Q to identity matrix
     Q = |1 0 .. 0 |
     |0 1 .. 0 |
     |0 .....  |
     */
    CHECK_CUDA( cudaMemset(mat_base, 0, internal_matrix_size * sizeof(real_t)));
    calculate_dimensions(m, n);
    
    set_ident <<< dim3(blks_wide_total), BLK_WIDTH >>> (mat_base, ld_panel_size, m, n);
    
    // Set "current" pointers
    mat_cur= mat_base;
    Q = Q_base;
    
    int k_blks = (n + BLK_WIDTH - 1) / BLK_WIDTH;
    
    // A bit of a hack, but it's probably fine.
    for(int panel = 0 ; panel < blks_wide_total - 1 ; panel++) increment(true);
    
    for(int panel = blks_wide_total - 1 ; panel >= 0 ; panel--) {
        
        // Probably want to iterate through "levels" here. That's why you used STL right?
        for (int lev = levels.size() - 1; lev > 0; lev--) {
            level_t * cur_lev = levels[lev];
            
            hh_update_triangle_reverse <<< dim3(cur_lev->num_blocks, k_blks), NUM_THREADS >>>
            (mat_cur, blockQ(lev), ld_panel_size, ldq, cur_lev->block_offset, mat_cur + m_current * BLK_WIDTH);
            
        }
        
        // Update two blocks on the left and right
        hh_update_dense_reverse <<< dim3(blks_tall_cur, k_blks), NUM_THREADS >>> (mat_cur, blockQ(0), ld_panel_size, ldq);
        
        // Next panel
        decrement(true);
    }
}

int QR_matrix::set_levels() {
    
    levels.clear();
    blks_tall_cur = ((m_current + BLK_HEIGHT - 1) / BLK_HEIGHT);
    blks_wide_cur = (n_current + BLK_WIDTH - 1) / BLK_WIDTH;
    int num_blocks = blks_tall_cur;
    int tb = num_blocks;
    
    // Add the first block
    level_t *lev = new level_t(num_blocks, 0, 0, 0);
    levels.push_back(lev);
    int block_offset = 1;
    int levnum = 1;
    
    while(num_blocks > 1) {
        num_blocks = (num_blocks + (BLK_HEIGHT/BLK_WIDTH) - 1) / (BLK_HEIGHT / BLK_WIDTH);
        lev = new level_t(num_blocks, tb, block_offset, levnum);
        levels.push_back(lev);
        block_offset *= (BLK_HEIGHT/BLK_WIDTH);
        levnum++;
        tb += num_blocks;
    }
    
    return tb;
    
}

void QR_matrix::panelTransInv(real_t * mat_out, const int m, const int n, const int lda) {
    
    dim3 gridDim(blks_tall_total, blks_wide_total);
    dim3 blockDim(BLK_HEIGHT);
    trans_inv <<< gridDim, blockDim >>> (mat_out, mat_base, ld_panel_size, m, n, lda);
    
}

void QR_matrix::retrieveR(real_t * mat_out, const int m, const int n, const int lda) {
    
    dim3 gridDim((blks_wide_total + BLK_ROWS - 1) /  BLK_ROWS , blks_wide_total);
    dim3 blockDim(BLK_HEIGHT);
    trans_inv <<< gridDim, blockDim >>> (mat_out, mat_base, ld_panel_size, m, n, lda);
    
}

// Add to "current" pointers by one panel
void QR_matrix::increment(bool levelChagneFlag) {
    
    if(!levelChagneFlag) mat_cur = mat_cur+ ld_panel_size;
    mat_cur = mat_cur+ BLK_WIDTH * BLK_WIDTH;
    Q = Q + ldq_panel;
    m_current -= BLK_WIDTH;
    n_current -= BLK_WIDTH;
    set_levels();
    
}

// Add to "current" pointers by one panel
void QR_matrix::decrement(bool levelChagneFlag) {
    if(!levelChagneFlag) mat_cur= mat_cur- ld_panel_size;
    mat_cur = mat_cur - BLK_WIDTH * BLK_WIDTH;
    Q = Q - ldq_panel;
    m_current += BLK_WIDTH;
    n_current += BLK_WIDTH;
    set_levels();
}

/* destructor */
QR_matrix::~QR_matrix() {
    CHECK_CUDA( cudaFree(mat_base));
    // CHECK_CUDA( cudaFree(Q_base));
}

level_t::level_t() {}

level_t::level_t(int nb, int ab, int bo, int l) {
    num_blocks = nb;
    aggregate_blocks = ab;
    block_offset = bo;
    lev = l;
}


void orth_CAQR(real_t *d_A, const uint64_t m, const uint64_t n){
    
    const int lda = roundup_to_32X( m );
    
    QR_matrix *QRobj = new QR_matrix(d_A, m, n, lda);
    
    // QR factorization
    QRobj->factor();
    //QRobj->retrieveR(d_A, m, n, lda);
    
    // Retrieve Q
    QRobj->retrieveQ();
    QRobj->panelTransInv(d_A, m, n, lda);
    
    CHECK_CUDA( cudaThreadSynchronize() );
    CHECK_CUDA( cudaGetLastError() );
    delete QRobj;
    
}
