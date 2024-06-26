#include "matrixMultiplyGPU.cuh"

#define TILESIZE_N  64
#define TILESIZE_K  16
#define THREADTILE   4

__host__ void matrixMultiply_GPU(int N, const float* A, const float* B, float* C, int *arg, int argCount)
{
    // NOTE: GradeBot already copies A, B and C to the GPU

    // Clear any existing values in C before calculating new values
    cudaMemset(C, 0, N * N * sizeof(float));

    // NOTE: N must be a multiple of TILESIZE_N
    dim3 numBlocks(N / TILESIZE_N, N / TILESIZE_N);
    dim3 threadsPerBlock(TILESIZE_N * TILESIZE_N / (THREADTILE * THREADTILE));
    matrixMultiplyKernel_GPU<<<numBlocks, threadsPerBlock>>>(N, A, B, C, 0, 0, 0);
}

__global__ void matrixMultiplyKernel_GPU(int N, const float* A, const float* B, float* C, int flag0, int flag1, int flag2)
{
    // NOTE: this implementation borrows from the 2D blocktiling explanation of:
    //
    //  S. Boehm. "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance:
    //      a Worklog." siboehm.com. https://siboehm.com/articles/22/CUDA-MMM
    //      (accessed Oct. 1, 2023).
    //
    // However, unlike Boehm, this kernel is specific to matrix-matrix multiply
    // (not a GEMM kernel) and uses 2D shared and thread-local memory. Further,
    // whereas Boehm's kernel operates on row-major matrices, this kernel is
    // intended for column-major matrices.

    // Indices of current thread within its block of 16x16 threads
    const int threadRow = threadIdx.x % (TILESIZE_N / THREADTILE);
    const int threadCol = threadIdx.x / (TILESIZE_N / THREADTILE);

    // Tiles of A and B cached in shared memory
    __shared__ float tileA[TILESIZE_K][TILESIZE_N];
    __shared__ float tileB[TILESIZE_N][TILESIZE_K];

    // Thread-local alloc. for each thread to calc. 4x4 set of elements of C
    float threadA[THREADTILE] = {0.0};
    float threadB[THREADTILE] = {0.0};
    float threadC[THREADTILE][THREADTILE] = {{0.0}};

    // Move pointers to A, B and C to relevant positions for this thread block
    A += blockIdx.y * TILESIZE_N;
    B += blockIdx.x * TILESIZE_N * N;
    C += (blockIdx.y + blockIdx.x * N) * TILESIZE_N;

    // Cache-loading addresses for this thread
    const int loadRowA = threadIdx.x % TILESIZE_N;
    const int loadColA = threadIdx.x / TILESIZE_N;
    const int colStrideA = blockDim.x / TILESIZE_N;

    const int loadRowB = threadIdx.x % TILESIZE_K;
    const int loadColB = threadIdx.x / TILESIZE_K;
    const int colStrideB = blockDim.x / TILESIZE_K;

    for (int k = 0; k < N; k += TILESIZE_K)
    {
        // Cache tiles from A and B into shared memory
        for (int colOffset = 0; colOffset < TILESIZE_K; colOffset += colStrideA)
        {
            tileA[loadColA + colOffset][loadRowA] =
                A[(loadColA + colOffset) * N + loadRowA];
        }
        for (int colOffset = 0; colOffset < TILESIZE_N; colOffset += colStrideB)
        {
            tileB[loadColB + colOffset][loadRowB] =
                B[(loadColB + colOffset) * N + loadRowB];
        }
        __syncthreads();

        // Advance pointers to A, B, and C to tiling positions for next iter.
        A += TILESIZE_K * N;
        B += TILESIZE_K;

        // Calculate elements of C that this thread is responsible for
        for (int tileK = 0; tileK < TILESIZE_K; tileK++)
        {
            // Load row of A and column of B into thread-local registers
            for (int threadI = 0; threadI < THREADTILE; threadI++)
            {
                threadA[threadI] = tileA[tileK][threadRow * THREADTILE + threadI];
            }
            for (int threadJ = 0; threadJ < THREADTILE; threadJ++)
            {
                threadB[threadJ] = tileB[threadCol * THREADTILE + threadJ][tileK];
            }
            // Accumulate matrix multiplication result in thread-local registers
            for (int threadJ = 0; threadJ < THREADTILE; threadJ++)
            {
                for (int threadI = 0; threadI < THREADTILE; threadI++)
                {
                    threadC[threadJ][threadI] += threadA[threadI] * threadB[threadJ];
                }
            }
        }
        __syncthreads();
    }

    // Update C with thread-local results
    const int threadBaseIdx = (threadRow + threadCol * N) * THREADTILE;
    for (int threadJ = 0; threadJ < THREADTILE; threadJ++)
    {
        for (int threadI = 0; threadI < THREADTILE; threadI++)
        {
            C[threadBaseIdx + threadI + threadJ * N] = threadC[threadJ][threadI];
        }
    }
}