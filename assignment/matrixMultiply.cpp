#include "matrixMultiply.h"

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

    // Cache blocking using block sizes of 64
    for (int j = 0; j < N; j += 64)
    {
        for (int k = 0; k < N; k += 64)
        {
            for (int i = 0; i < N; i += 64)
            {
                #pragma omp parallel for
                for (int jj = 0; jj < 64; jj++)
                {
                    int J = j + jj;
                    for (int kk = 0; kk < 64; kk++)
                    {
                        int K = k + kk;
                        __m256 b = _mm256_set1_ps(B[K + J * N]);
                        for (int ii = 0; ii < 64; ii += 8)
                        {
                            int I = i + ii;
                            // Fortran order
                            __m256 a = _mm256_loadu_ps(A + I + K * N);
                            __m256 c = _mm256_loadu_ps(C + I + J * N);
                            __m256 x = _mm256_mul_ps(a, b);
                            c = _mm256_add_ps(c, x);
                            _mm256_storeu_ps(C + I + J * N, c);
                        }
                    }
                }
            }
        }
    }
}