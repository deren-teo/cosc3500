#include "gridEvolveGPU.cuh"

#define cudaCheck(expr) \
    do { \
        cudaError_t e = (expr); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            abort(); \
        } \
    } while (false)

// Given a byte representing a cell state and neighbour sum in the form:
//     <3 bits: unused><4 bits: neighbour sum><1 bit: state>
// this value maps the byte to 1 if the state will change, else 0
#define TRANSITION_MAP 0x2AA4A  // bin: 101010101001001010


__host__ void gridEvolve_GPU(char *grid, char *temp, const int nRows, const int nCols, char *isStatic)
{
    // NOTE: main script handles all cudaMemcpy'ing (before, during and after)

    // Run kernel
    gridEvolveKernel_GPU<<<1, 1>>>(grid, temp, nRows, nCols, isStatic);
}

__global__ void gridEvolveKernel_GPU(char *grid, char *temp, const int nRows, const int nCols, char *isStatic)
{
    const int rowSize = nCols + 2;

    // Update cell states in grid copy
    *isStatic = 1;
    int idx = 0;
    for (int i = 0; i < nRows; i++) {
        idx += rowSize;
        for (int j = 1; j <= nCols; j++) {
            int idxj = idx + j;
            if (TRANSITION_MAP & (1 << grid[idxj])) {
                const int idx_abv = idxj - rowSize;
                const int idx_blw = idxj + rowSize;
                if (grid[idxj] & 0x01) {
                    temp[idx_abv - 1] -= 2;
                    temp[idx_abv]     -= 2;
                    temp[idx_abv + 1] -= 2;
                    temp[idxj - 1]    -= 2;
                    temp[idxj]        &= 0xFE;
                    temp[idxj + 1]    -= 2;
                    temp[idx_blw - 1] -= 2;
                    temp[idx_blw]     -= 2;
                    temp[idx_blw + 1] -= 2;
                } else {
                    temp[idx_abv - 1] += 2;
                    temp[idx_abv]     += 2;
                    temp[idx_abv + 1] += 2;
                    temp[idxj - 1]    += 2;
                    temp[idxj]        |= 0x01;
                    temp[idxj + 1]    += 2;
                    temp[idx_blw - 1] += 2;
                    temp[idx_blw]     += 2;
                    temp[idx_blw + 1] += 2;
                }
                *isStatic = 0;
            }
        }
    }
}
