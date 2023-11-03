#include "gridEvolveGPU.cuh"

#define cudaCheck(expr) \
    do { \
        cudaError_t e = (expr); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            abort(); \
        } \
    } while (false)

#define BLOCKTILE  64
#define THREADTILE  4

// Given a byte representing a cell state and neighbour sum in the form:
//     <3 bits: unused><4 bits: neighbour sum><1 bit: state>
// this value maps the byte to 1 if the state will change, else 0
#define TRANSITION_MAP 0x2AA4A  // bin: 101010101001001010


__host__ void gridEvolve_GPU(char *grid, char *temp, const int nRows, const int nCols, char *isStatic)
{
    // NOTE: main script handles all cudaMemcpy'ing (before, during and after)

    // // Initially set isStatic as true so any thread can set it false
    // *isStatic = 1;

    // NOTE: nCols and nRows must both be a multiple of BLOCKTILE
    dim3 numBlocks(nCols / BLOCKTILE, nRows / BLOCKTILE);
    dim3 threadsPerBlock(BLOCKTILE * BLOCKTILE / (THREADTILE * THREADTILE));
    gridEvolveKernel_GPU<<<numBlocks, threadsPerBlock>>>(grid, temp, nRows, nCols, isStatic);
}

__global__ void gridEvolveKernel_GPU(char *grid, char *temp, const int nRows, const int nCols, char *isStatic)
{
    const int rowSize = nCols + 2;

    // Indices of current thread within its block
    const int threadRow = threadIdx.x / (BLOCKTILE / THREADTILE);
    const int threadCol = threadIdx.x % (BLOCKTILE / THREADTILE);

    // Move pointers to grids to relevant positions for this thread block,
    // (... + rowSize + 1) to account for zero-padding
    const int blockStartIdx = (blockIdx.y * BLOCKTILE) * rowSize +
                              (blockIdx.x * BLOCKTILE) + rowSize + 1;
    grid += blockStartIdx;
    temp += blockStartIdx;

    // // Define local equivalent to avoid unnecessary global memory accesses
    // char locallyStatic = 1;

    // Thread to address cells that it is responsible for
    for (int row = 0; row < THREADTILE; row++)
    {
        for (int col = 0; col < THREADTILE; col++)
        {
            const int idx = (threadRow * THREADTILE + row) * rowSize +
                             threadCol * THREADTILE + col;
            const int state = grid[idx];

            // Determine neighbourhood sum
            const int idx_abv = idx - rowSize;
            const int idx_blw = idx + rowSize;
            char neighbourSum = 0;
            neighbourSum += grid[idx_abv - 1];
            neighbourSum += grid[idx_abv];
            neighbourSum += grid[idx_abv + 1];
            neighbourSum += grid[idx - 1];
            neighbourSum += grid[idx + 1];
            neighbourSum += grid[idx_blw - 1];
            neighbourSum += grid[idx_blw];
            neighbourSum += grid[idx_blw + 1];

            // TODO: optimise by loading all cells that need to be checked
            //  (i.e. 6x6 bytes) into cache. Can also go a step further and load
            //  all cells needed by a block into shared memory. Actually, might
            //  be better trying this vice versa. This might also lend itself
            //  well to not needing a temp grid at all.

            // Lookup if state changes
            if (TRANSITION_MAP & (1 << ((neighbourSum << 1) | state)))
            {
                temp[idx] = !state;
                // locallyStatic = 0;
            }
        }
    }

    // // Update grid static status (multiple threads trying to write doesn't
    // // matter as long as at least one manages to write)
    // if (!locallyStatic)
    // {
    //     *isStatic = 0;
    // }
}
