#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

/**
 * @brief Updates the Life grid one timestep based on the standard rules
 *
 * @param[in] grid Pointer to device memory containing main grid
 * @param[in] temp Pointer to device memory containing intermediate grid
 * @param[in] nRows Number of grid rows
 * @param[in] nCols Number of grid columns
 * @param[out] isStatic Pointer to static flag on host memory
 * @return void
*/
__host__ void gridEvolve_GPU(char *grid, char *temp, const int nRows, const int nCols, char *isStatic);

/**
 * @brief Kernel handles most of the computation of the the above in parallel
 *
 * @param[in] grid Pointer to device memory containing main grid
 * @param[in] temp Pointer to device memory containing intermediate grid
 * @param[in] nRows Number of grid rows
 * @param[in] nCols Number of grid columns
 * @param[out] gridStatic Pointer to device memory containing per-block static flags
 * @return void
*/
__global__ void gridEvolveKernel_GPU(char *grid, char *temp, const int nRows, const int nCols, char *gridStatic);
