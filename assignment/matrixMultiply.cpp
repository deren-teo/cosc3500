#include "matrixMultiply.h"
// #include <cstdio> // DEBUG
/**
* @brief Implements an NxN matrix multiply C=A*B
*
* @param[in] N : dimension of square matrix (NxN)
* @param[in] A : pointer to input NxN matrix
* @param[in] B : pointer to input NxN matrix
* @param[out] C : pointer to output NxN matrix
* @param[in] args : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If unused, set to zero
* @param[in] argCount : the length of the flags array
* @return void
* */
void matrixMultiply(int N, const float* A, const float* B, float* C, int* args, int argCount)
{
    // Clear any existing values in C before calculating new values
    memset(C, 0, sizeof(float) * N * N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                // Fortran order
                C[i + j * N] += A[i + k * N] * B[k + j * N];
            }
        }
    }
}