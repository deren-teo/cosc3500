#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

/**
 * @brief Updates the Life grid one timestep based on the standard rules
 *
 * @param[in] grid
 * @param[out] isStatic
 * @return void
*/
__host__ void gridEvolve_GPU(char *grid, char *temp, const int nRows, const int nCols, char *isStatic);

/**
 * @brief Kernel handles most of the computation of the the above in parallel
 *
 * @param[in] grid
 * @param[out] isStatic
 * @return void
*/
__global__ void gridEvolveKernel_GPU(char *grid, char *temp, const int nRows, const int nCols, char *isStatic);