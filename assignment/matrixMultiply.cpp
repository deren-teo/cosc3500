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
    #pragma omp parallel for
    for (int j = 0; j < N; j += 64)
    {
        for (int k = 0; k < N; k += 64)
        {
            for (int i = 0; i < N; i += 64)
            {
                for (int jj = 0; jj < 64; jj++)
                {
                    int J = j + jj;
                    for (int kk = 0; kk < 64; kk++)
                    {
                        int K = k + kk;
                        __m256 b = _mm256_set1_ps(B[K + J * N]);

                        // Fortran order
                        __m256 a1 = _mm256_load_ps(A + i + K * N);
                        __m256 a2 = _mm256_load_ps(A + i + 8 + K * N);
                        __m256 a3 = _mm256_load_ps(A + i + 16 + K * N);
                        __m256 a4 = _mm256_load_ps(A + i + 24 + K * N);
                        __m256 a5 = _mm256_load_ps(A + i + 32 + K * N);
                        __m256 a6 = _mm256_load_ps(A + i + 40 + K * N);
                        __m256 a7 = _mm256_load_ps(A + i + 48 + K * N);
                        __m256 a8 = _mm256_load_ps(A + i + 56 + K * N);

                        __m256 c1 = _mm256_load_ps(C + i + J * N);
                        __m256 c2 = _mm256_load_ps(C + i + 8 + J * N);
                        __m256 c3 = _mm256_load_ps(C + i + 16 + J * N);
                        __m256 c4 = _mm256_load_ps(C + i + 24 + J * N);
                        __m256 c5 = _mm256_load_ps(C + i + 32 + J * N);
                        __m256 c6 = _mm256_load_ps(C + i + 40 + J * N);
                        __m256 c7 = _mm256_load_ps(C + i + 48 + J * N);
                        __m256 c8 = _mm256_load_ps(C + i + 56 + J * N);

                        __m256 x1 = _mm256_mul_ps(a1, b);
                        __m256 x2 = _mm256_mul_ps(a2, b);
                        __m256 x3 = _mm256_mul_ps(a3, b);
                        __m256 x4 = _mm256_mul_ps(a4, b);
                        __m256 x5 = _mm256_mul_ps(a5, b);
                        __m256 x6 = _mm256_mul_ps(a6, b);
                        __m256 x7 = _mm256_mul_ps(a7, b);
                        __m256 x8 = _mm256_mul_ps(a8, b);

                        c1 = _mm256_add_ps(c1, x1);
                        c2 = _mm256_add_ps(c2, x2);
                        c3 = _mm256_add_ps(c3, x3);
                        c4 = _mm256_add_ps(c4, x4);
                        c5 = _mm256_add_ps(c5, x5);
                        c6 = _mm256_add_ps(c6, x6);
                        c7 = _mm256_add_ps(c7, x7);
                        c8 = _mm256_add_ps(c8, x8);

                        _mm256_store_ps(C + i + J * N, c1);
                        _mm256_store_ps(C + i + 8 + J * N, c2);
                        _mm256_store_ps(C + i + 16 + J * N, c3);
                        _mm256_store_ps(C + i + 24 + J * N, c4);
                        _mm256_store_ps(C + i + 32 + J * N, c5);
                        _mm256_store_ps(C + i + 40 + J * N, c6);
                        _mm256_store_ps(C + i + 48 + J * N, c7);
                        _mm256_store_ps(C + i + 56 + J * N, c8);
                    }
                }
            }
        }
    }
}