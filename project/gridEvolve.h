#include <vector>

/**
 * @brief Updates the grid one timestep  based on the standard rules of Life.
 *
 * @param[in] grid Pointer to the memory allocated for the grid
 * @param[in] nRows Number of grid rows, not including zero-padding
 * @param[in] nCols Number of grid columns, not including zero-padding
 * @param[out] isStatic Flag indicating if grid has reached a static state
 * @return void
*/
void gridEvolve(char *grid, const int nRows, const int nCols, int *isStatic);
