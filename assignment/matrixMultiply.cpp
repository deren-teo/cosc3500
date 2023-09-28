#include "matrixMultiply.h"

#define N_BLOCK 64 // cache blocking occurs in blocks of size N_BLOCK x N_BLOCK

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

    // Cache blocking using block sizes of N_BLOCK
    #pragma omp parallel for
    for (int j = 0; j < N; j += N_BLOCK)
    {
        for (int k = 0; k < N; k += N_BLOCK)
        {
            for (int i = 0; i < N; i += N_BLOCK)
            {
                for (int jj = 0; jj < N_BLOCK; jj++)
                {
                    int J = j + jj;
                    for (int kk = 0; kk < N_BLOCK; kk++)
                    {
                        int K = k + kk;
                        // Fortran order
                        __m256 b = _mm256_set1_ps(B[K + J * N]);

                        const float *addrBaseA = A + i + K * N;
                        float *addrBaseC = C + i + J * N;

                        // Innermost unrolled loop has cache block size hardcoded
                        __m256 a1 = _mm256_load_ps(addrBaseA);
                        __m256 a2 = _mm256_load_ps(addrBaseA + 8);
                        __m256 a3 = _mm256_load_ps(addrBaseA + 16);
                        __m256 a4 = _mm256_load_ps(addrBaseA + 24);
                        __m256 a5 = _mm256_load_ps(addrBaseA + 32);
                        __m256 a6 = _mm256_load_ps(addrBaseA + 40);
                        __m256 a7 = _mm256_load_ps(addrBaseA + 48);
                        __m256 a8 = _mm256_load_ps(addrBaseA + 56);

                        __m256 c1 = _mm256_load_ps(addrBaseC);
                        __m256 c2 = _mm256_load_ps(addrBaseC + 8);
                        __m256 c3 = _mm256_load_ps(addrBaseC + 16);
                        __m256 c4 = _mm256_load_ps(addrBaseC + 24);
                        __m256 c5 = _mm256_load_ps(addrBaseC + 32);
                        __m256 c6 = _mm256_load_ps(addrBaseC + 40);
                        __m256 c7 = _mm256_load_ps(addrBaseC + 48);
                        __m256 c8 = _mm256_load_ps(addrBaseC + 56);

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

                        _mm256_store_ps(addrBaseC, c1);
                        _mm256_store_ps(addrBaseC + 8, c2);
                        _mm256_store_ps(addrBaseC + 16, c3);
                        _mm256_store_ps(addrBaseC + 24, c4);
                        _mm256_store_ps(addrBaseC + 32, c5);
                        _mm256_store_ps(addrBaseC + 40, c6);
                        _mm256_store_ps(addrBaseC + 48, c7);
                        _mm256_store_ps(addrBaseC + 56, c8);
                    }
                }
            }
        }
    }
}