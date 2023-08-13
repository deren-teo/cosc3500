/*******************************************************************************
 * @file    main.cpp
 * @author  Deren Teo
 * @brief   An optimised implementation of Conway's Game of Life (abbr. Life).
 *      NOTE: This source file kinda conforms with the Google C++ Style Guide.
 ******************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cstring>

/**
 * Allocates memory for a 2D grid on which Life will evolve.
 *
 * @param n_rows Number of rows in the grid
 * @param n_cols Number of columns in the grid
 *
 * @return Pointer to the memory allocated for the grid.
*/
static uint8_t *GridCreate(const int n_rows, const int n_cols) {
    return (uint8_t *)std::malloc(n_rows * n_cols * sizeof(uint8_t));
}

/**
 * Returns the sum of the 9 cells surrounding and including (row, col).
 *
 * @param grid Pointer to memory allocated for the grid
 * @param row Row number of centre cell
 * @param col Column number of centre cell
 * @param n_cols Number of columns in the grid
 *
 * @return Sum of the 3x3 grid of cells centred on (row, col).
*/
static int GridLocalSum(uint8_t *grid, const int row, const int col,
                        const int n_cols) {
    int sum = 0;
    sum += static_cast<int>(grid[(row - 1) * n_cols + (col - 1)]);
    sum += static_cast<int>(grid[(row - 1) * n_cols + (col)    ]);
    sum += static_cast<int>(grid[(row - 1) * n_cols + (col + 1)]);
    sum += static_cast<int>(grid[(row)     * n_cols + (col - 1)]);
    sum += static_cast<int>(grid[(row)     * n_cols + (col)    ]);
    sum += static_cast<int>(grid[(row)     * n_cols + (col + 1)]);
    sum += static_cast<int>(grid[(row + 1) * n_cols + (col - 1)]);
    sum += static_cast<int>(grid[(row + 1) * n_cols + (col)    ]);
    sum += static_cast<int>(grid[(row + 1) * n_cols + (col + 1)]);
    return sum;
}

/**
 * Get the binary state of the cell (i, j).
 *
 * @param grid Pointer to memory allocated for the grid
 * @param row Row number of the cell
 * @param col Column number of the cell
 * @param n_cols Number of columns in the grid
 *
 * @return 1 for alive, 0 for dead.
*/
static inline char GridGetState(uint8_t *grid, const int row, const int col,
                                const int n_cols) {
    return grid[row * n_cols + col];
}

/**
 * Set the binary state of the cell (i, j).
 *
 * @param grid Pointer to memory allocated for the grid
 * @param row Row number of the cell
 * @param col Column number of the cell
 * @param n_cols Number of columns in the grid
*/
static inline void GridSetState(uint8_t *grid, const int row, const int col,
                                const int n_cols, uint8_t state) {
    grid[row * n_cols + col] = state;
}

/**
 * Updates the grid one timestep forward based on the standard rules of Life.
 * Cells beyond the grid are considered to be dead.
 *
 * @param grid Pointer to memory allocated for the grid
 * @param n_rows Number of rows in the grid
 * @param n_cols Number of columns in the grid
 *
 * @return 1 if all cells in the grid are dead (boolean false).
*/
static int GridEvolve(uint8_t *grid, const int n_rows, const int n_cols) {
    // Allocate copy of grid to hold intermediate evolved cell states
    uint8_t *evolved_grid = static_cast<uint8_t *>(
        std::malloc(n_rows * n_cols * sizeof(uint8_t)));
    std::memcpy(evolved_grid, grid, n_rows * n_cols * sizeof(uint8_t));

    // Keep track of whether any cells are alive
    bool any_alive = false;

    // Update cell states in grid copy
    for (unsigned short i = 0; i < n_rows; i++) {
        for (unsigned short j = 0; j < n_cols; j++) {
            // TODO: should probably use a 3x3 kernel
            switch (GridLocalSum(grid, i, j, n_cols)) {
                case 3: {
                    GridSetState(grid, i, j, n_cols, 1);
                    any_alive = true;
                    break;
                }
                case 4: {
                    any_alive |= (GridGetState(grid, i, j, n_cols) == 1);
                    break;
                }
                default: {
                    GridSetState(grid, i, j, n_cols, 0);
                    break;
                }
            }
        }
    }
    // Copy evolved cell states back into grid and free temporary grid copy
    std::memcpy(grid, evolved_grid, n_rows * n_cols * sizeof(uint8_t));
    free(evolved_grid);

    return static_cast<int>(any_alive);
}

int main(int argc, char *argv[]) {
    // TODO: un-hardcode these
    const int kGridRows = 10;
    const int kGridCols = 10;
    const int kSimIterations = 100;

    // Create the grid
    uint8_t *grid = GridCreate(kGridRows, kGridCols);

    // TODO: initial configuration

    // Evolve the simulation the specified number of iterations or until all
    // cells are dead (meaning nothing will happen in all future iterations)
    for (int i = 0; i < kSimIterations; i++) {
        if (GridEvolve(grid, kGridRows, kGridCols) == 1) {
            break;
        }
    }

    // Clean up and exit
    free(grid);
    return 0;
}
