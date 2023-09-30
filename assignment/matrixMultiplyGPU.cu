#include "matrixMultiplyGPU.cuh"
#include <iostream> // DEBUG only

#define BLOCKSIZE 32 // cache blocking occurs in blocks of size BLOCKSIZE x BLOCKSIZE

// DEBUG only
void checkError(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
        abort();
    }
}

__host__ void matrixMultiply_GPU(int N, const float* A, const float* B, float* C, int *arg, int argCount)
{
    // NOTE: GradeBot already copies A, B and C to the GPU

    // Clear any existing values in C before calculating new values
    checkError(cudaMemset(C, 0, N * N * sizeof(float)));

    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    matrixMultiplyKernel_GPU<<<numBlocks, threadsPerBlock>>>(N, A, B, C, 0, 0, 0);
}

__global__ void matrixMultiplyKernel_GPU(int N, const float* A, const float* B, float* C, int flag0, int flag1, int flag2)
{
    __shared__ float blockA[BLOCKSIZE][BLOCKSIZE]; // SMEM caches
    __shared__ float blockB[BLOCKSIZE][BLOCKSIZE];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    float cij = 0;
    for (int blockNum = 0; blockNum < gridDim.x; blockNum++)
    {
        // Each thread caches one element of a block of A and B each into SMEM
        blockA[threadIdx.y][threadIdx.x] = A[i + (blockNum * BLOCKSIZE + threadIdx.x) * N];
        blockB[threadIdx.y][threadIdx.x] = B[(blockNum * BLOCKSIZE + threadIdx.y) + j * N];
        __syncthreads();

        // Each thread calculates the vector dot product for one element in C
        for (int k = 0; k < BLOCKSIZE; k++)
        {
            cij += blockA[threadIdx.y][k] * blockB[k][threadIdx.x];
        }
        __syncthreads();

        C[i + j * N] = cij;
    }
}