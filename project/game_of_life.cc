/*******************************************************************************
 * @file    main.cc
 * @author  Deren Teo
 * @brief   An optimised implementation of Conway's Game of Life (abbr. Life).
 ******************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "parser.h"

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
 * Randomly initialise each cell in the grid as either alive (1) or dead (0).
 * If not manually seeded, std::rand() behaves as if seeded with std::srand(1).
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_rows Number of rows in the grid
 * @param n_cols Number of columns in the grid
*/
static void GridRandomInit(uint8_t *grid, const int n_rows, const int n_cols) {
    for (int i = 0; i < n_rows * n_cols; i++) {
        grid[i] = static_cast<uint8_t>(std::rand() & 0x01);
    }
}

/**
 * Returns the sum of the 3x3 grid of cells aronud and including (row, col).
 * If row or col is on the edge of the grid, the 3x3 grid wraps around to the
 * other side of the grid, as if on a taurus.
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param row Row number of centre cell
 * @param col Column number of centre cell
 * @param n_cols Number of columns in the grid
 *
 * @return Sum of the 3x3 grid of cells centred on (row, col).
*/
static int GridLocalSum(uint8_t *grid, const int row, const int col,
                        const int n_rows, const int n_cols) {
    // Surrounding row and column indices with wraparound logic
    int row0 = (row - 1 + n_rows) % n_rows;
    int row2 = (row + 1) % n_rows;
    int col0 = (col - 1 + n_cols) % n_cols;
    int col2 = (col + 1) % n_cols;

    int sum = 0;
    sum += (grid[row0 * n_cols + col0]);
    sum += (grid[row0 * n_cols + col ]);
    sum += (grid[row0 * n_cols + col2]);
    sum += (grid[row  * n_cols + col0]);
    sum += (grid[row  * n_cols + col ]);
    sum += (grid[row  * n_cols + col2]);
    sum += (grid[row2 * n_cols + col0]);
    sum += (grid[row2 * n_cols + col ]);
    sum += (grid[row2 * n_cols + col2]);
    return sum;
}

/**
 * Get the binary state of the cell (i, j).
 *
 * @param grid Pointer to the memory allocated for the grid
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
 * @param grid Pointer to the memory allocated for the grid
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
 *
 * @param grid Pointer to the memory allocated for the grid
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
            switch (GridLocalSum(grid, i, j, n_rows, n_cols)) {
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
    // Default simulation configuration
    int n_rows = 10;
    int n_cols = 10;
    int n_iter = 100;
    uint8_t *grid;

    // TODO: this command line argument parsing is not very robust
    switch (argc) {
        case 1: {
            grid = GridCreate(n_rows, n_cols);
            GridRandomInit(grid, n_rows, n_cols);
            break;
        }
        case 2: {
            n_iter = std::atoi(argv[1]);
            grid = GridCreate(n_rows, n_cols);
            GridRandomInit(grid, n_rows, n_cols);
            break;
        }
        case 3: {
            n_iter = std::atoi(argv[1]);
            grid = ParseRLEFile(argv[2]);
            break;
        }
        default: {
            return 1;
        }
    }

    // Evolve the simulation the specified number of iterations or until all
    // cells are dead (meaning nothing will happen in all future iterations)
    for (int i = 0; i < n_iter; i++) {
        if (GridEvolve(grid, n_rows, n_cols) == 1) {
            break;
        }
    }

    // Dump results to console
    for (int i = 0; i < n_rows; i++) {
        std::cout << "|";
        for (int j = 0; j < n_cols; j++) {
            std::cout << static_cast<int>(grid[i * n_cols + j]) << "|";
        }
        std::cout << "\n";
    }

    // Clean up and exit
    free(grid);
    return 0;
}
