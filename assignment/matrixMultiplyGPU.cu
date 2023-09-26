#include "matrixMultiplyGPU.cuh"
#include <iostream> // DEBUG only

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

    int threadsPerBlock = 1024;
    int numBlocks = N * N / threadsPerBlock;
    matrixMultiplyKernel_GPU<<<numBlocks, threadsPerBlock>>>(N, A, B, C, 0, 0, 0);
}

__global__ void matrixMultiplyKernel_GPU(int N, const float* A, const float* B, float* C, int flag0, int flag1, int flag2)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int i = z % N;
    int j = z / N;
    for (int k = 0; k < N; k++)
    {
        C[z] += A[i + k * N] * B[k + j * N];
    }
}