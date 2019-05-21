# Randomized_SVD_in_CUDA

This is an implementation of [Randomized SVD by Halko et al.](https://arxiv.org/abs/0909.4061) for CPU and GPU. 

# Approach

There are various ways for computing SVD. The standard methods are the Gram-Schmidt process, House-holder reflections, Givens rotations. 
These numerical methods has been proven to be very accurate. 
However in the realm of Collaborative Filtering or PCA, one might not be interested in computing full SVD, but just the few largest singular values. Given the need for less accurate, but more efficient solution, in 2009 Halko et al. proposed a probabilistic approach. 
Halko argued that "these methods use random sampling to identify a subspace that captures most of the action of a matrix. The input matrix is then compressed—either explicitly or implicitly—to this subspace, and the reduced matrix is manipulated deterministically to obtain the desired low-rank factorization.” 
Even though Halko showed that the randomized approach beats its classical competitors in terms of accuracy, speed, and robustness, the he did not investigate an implementation for a GPU. 


Previous works have shown the Householder algorithm is the most suitable algorithm for QR factorization on a GPU, because it is more stable than Gram-Schmidt, less expensive than Givens and good for parallelization on a GPU (Cosnuau 2014).
Other works have also shown that computing SVD through QR decomposition using standard methods for small matrices on a GPU does is less effectient then performing it on a CPU. (Ji et al)
In meantime Anderson M. et al proposed an GPU-Friendly QR decomposition algorithm built specifically  for tall and skinny or wide and short matrix, called Communication-Avoiding QR Decomposition (CAQR). Anderson M. et al show ed that the reduction in memory traffic provided by CAQR allows us to outperform existing parallel GPU implementations of QR for a large class of tall-skinny matrices. Furthermore they argus that their QR factorization is done entirely on the GPU using compute-bound kernels, meaning performance is good regardless of the width of the matrix. As a result, the algorithm outperforms CULA, a parallel linear algebra library for GPUs by up to 17x for tall-skinny matrices and Intel’s Math Kernel Library (MKL) by up to 12x. CAQR is an extension of Tall-Skinny QR (TSQR) for arbitrarily sized matrices. 

# Referances:

* [Randomized SVD by Halko N. et al](https://arxiv.org/abs/0909.4061)
* [CAQR by Anderson M. et al](http://www.netlib.org/lapack/lawnspdf/lawn240.pdf)
* [GPU Accelerated Randomized Singular Value Decomposition and Its Application in Image Compression by Ji et al](https://pdfs.semanticscholar.org/256d/46471cf2ceaaa999d73879e28c7f8d0854c3.pdf)
