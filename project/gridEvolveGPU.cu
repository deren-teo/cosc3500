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

    // NOTE: nCols and nRows must both be a multiple of BLOCKTILE
    dim3 numBlocks(nCols / BLOCKTILE, nRows / BLOCKTILE);
    dim3 threadsPerBlock(BLOCKTILE * BLOCKTILE / (THREADTILE * THREADTILE));

    // Allocate an array for each block to report "isStatic" status
    const int totalBlocks = numBlocks.x * numBlocks.y;
    char *gridStatic_GPU;
    cudaCheck(cudaMalloc((void **)&gridStatic_GPU, totalBlocks));

    // Execute kernel
    gridEvolveKernel_GPU<<<numBlocks, threadsPerBlock>>>(grid, temp, nRows, nCols, gridStatic_GPU);

    // Loop over all blocks to update overall "isStatic", returning early if possible
    char *gridStatic = (char *)std::malloc(totalBlocks);
    cudaCheck(cudaMemcpy(gridStatic, gridStatic_GPU, totalBlocks, cudaMemcpyDeviceToHost));
    *isStatic = 1;
    for (int i = 0; i < totalBlocks; i++)
    {
        if (!gridStatic[i])
        {
            *isStatic = 0;
            return;
        }
    }
}

__global__ void gridEvolveKernel_GPU(char *grid, char *temp, const int nRows, const int nCols, char *gridStatic)
{
    // Row size, accounting for grid zero-padding
    const int rowSize = nCols + 2;

    // Shared memory to store "isStatic" result for each thread in this block
    __shared__ char blockStatic[BLOCKTILE * BLOCKTILE / (THREADTILE * THREADTILE)];
    // Each thread initialises its own entry to 1
    blockStatic[threadIdx.x] = 1;

    // Indices of current thread within its block
    const int threadRow = threadIdx.x / (BLOCKTILE / THREADTILE);
    const int threadCol = threadIdx.x % (BLOCKTILE / THREADTILE);

    // Move pointers to grids to relevant positions for this thread block,
    // (... + rowSize + 1) to account for zero-padding
    const int blockStartIdx = (blockIdx.y * BLOCKTILE) * rowSize +
                              (blockIdx.x * BLOCKTILE) + rowSize + 1;
    grid += blockStartIdx;
    temp += blockStartIdx;

    // Allocate thread-local memory for all cells to be addressed by this thread
    char threadGrid[THREADTILE + 2][THREADTILE + 2];

    // Each thread to cache its own cells into thread-local registers
    for (int row = -1; row < THREADTILE + 1; row++)
    {
        for (int col = -1; col < THREADTILE + 1; col++)
        {
            const int gridIdx = (threadRow * THREADTILE + row) * rowSize +
                                (threadCol * THREADTILE + col);
            threadGrid[row + 1][col + 1] = grid[gridIdx];
        }
    }
    // NOTE: no thread synchronisation needed as grid is not modified until
    //  after the kernel returns

    // Each thread to update state of cells that it is responsible for
    for (int row = 1; row < THREADTILE + 1; row++)
    {
        for (int col = 1; col < THREADTILE + 1; col++)
        {
            // Determine neigbourhood sum
            char neighbourSum = 0;
            neighbourSum += threadGrid[row - 1][col - 1];
            neighbourSum += threadGrid[row - 1][col];
            neighbourSum += threadGrid[row - 1][col + 1];
            neighbourSum += threadGrid[row][col - 1];
            neighbourSum += threadGrid[row][col + 1];
            neighbourSum += threadGrid[row + 1][col - 1];
            neighbourSum += threadGrid[row + 1][col];
            neighbourSum += threadGrid[row + 1][col + 1];

            // Lookup if state changes
            const int state = threadGrid[row][col];
            if (TRANSITION_MAP & (1 << ((neighbourSum << 1) | state)))
            {
                const int gridIdx = (threadRow * THREADTILE + row - 1) * rowSize +
                                    (threadCol * THREADTILE + col - 1);
                temp[gridIdx] = !state;
                blockStatic[threadIdx.x] = 0;
            }
        }
    }

    // Reduction over shared memory to determine block "isStatic" status
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            blockStatic[threadIdx.x] &= blockStatic[threadIdx.x + i];
        }
    }
    if (threadIdx.x == 0)
    {
        gridStatic[blockIdx.x * blockDim.x + blockIdx.y] = blockStatic[0];
    }
}
