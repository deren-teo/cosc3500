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
 * Cells outside the grid are treated as zero (dead).
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param row Row number of centre cell
 * @param col Column number of centre cell
 * @param n_rows Number of rows in the grid
 * @param n_cols Number of columns in the grid
 *
 * @return Sum of the 3x3 grid of cells centred on (row, col).
*/
static int GridLocalSum(uint8_t *grid, const int row, const int col,
                        const int n_rows, const int n_cols) {
    int sum = 0;
    int idx00 = (row - 1) * n_cols + (col - 1);
    int idx10 = idx00 + n_cols;
    int idx20 = idx10 + n_cols;
    if (row > 0) {
        if (col > 0) {
            sum += grid[idx00];
        }
        sum += grid[idx00 + 1];
        if (col < (n_cols - 1)) {
            sum += grid[idx00 + 2];
        }
    }
    if (col > 0) {
        sum += grid[idx10];
    }
    sum += grid[idx10 + 1];
    if (col < (n_cols - 1)) {
        sum += grid[idx10 + 2];
    }
    if (row < (n_rows - 1)) {
        if (col > 0) {
            sum += grid[idx20];
        }
        sum += grid[idx20 + 1];
        if (col < (n_cols - 1)) {
            sum += grid[idx20 + 2];
        }
    }
    std::cout << sum;
    return sum;
}

/**
 * Set the binary state of the cell (i, j).
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param row Row number of the cell
 * @param col Column number of the cell
 * @param n_cols Number of columns in the grid
 * @param state_changed Pointer to variable to whether state has just changed
*/
static void CellSetState(uint8_t *grid, const int row, const int col,
        const int n_cols, uint8_t state, int *state_changed) {
    int idx = row * n_cols + col;
    *state_changed = (grid[idx] != state);
    grid[idx] = state;
}

/**
 * Updates the grid one timestep forward based on the standard rules of Life.
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_rows Number of rows in the grid
 * @param n_cols Number of columns in the grid
 *
 * @return 0 if successful. 1 if no cell states changed (signals early stop).
*/
static int GridEvolve(uint8_t *grid, const int n_rows, const int n_cols) {
    // Allocate copy of grid to hold intermediate evolved cell states
    uint8_t *evolved_grid = static_cast<uint8_t *>(
        std::malloc(n_rows * n_cols * sizeof(uint8_t)));
    std::memcpy(evolved_grid, grid, n_rows * n_cols * sizeof(uint8_t));

    // Early stopping condition: no cells changed (possibly all dead)
    int stop_early = 1;

    // Update cell states in grid copy
    int state_changed;
    for (unsigned short i = 0; i < n_rows; i++) {
        for (unsigned short j = 0; j < n_cols; j++) {
            // TODO: use a 3x3 kernel
            switch (GridLocalSum(grid, i, j, n_rows, n_cols)) {
                case 0: {
                    // All cells around (i, j) are dead; possibly early stop
                    break;
                }
                case 3: {
                    // Cell (i, j) now alive, regardless of previous state
                    CellSetState(grid, i, j, n_cols, 1, &state_changed);
                    stop_early &= !state_changed;
                    break;
                }
                case 4: {
                    // (i, j) does not change state; possible early stop
                    break;
                }
                default: {
                    // Cell (i, j) now dead, regardless of previous state
                    CellSetState(grid, i, j, n_cols, 0, &state_changed);
                    stop_early &= !state_changed;
                    break;
                }
            }
        }
    }
    // Copy evolved cell states back into grid and free temporary grid copy
    // TODO: highly memory intensive; is there any way to point `grid` at
    //  `evolved_grid` and free the old grid?
    std::memcpy(grid, evolved_grid, n_rows * n_cols * sizeof(uint8_t));
    free(evolved_grid);
    return stop_early;
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
            grid = ParseRLEFile(argv[2], &n_rows, &n_cols);
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
            uint8_t cell_state = grid[i * n_cols + j];
            if (cell_state) {
                std::cout << '#' << '|';
            } else {
                std::cout << ' ' << '|';
            }
        }
        std::cout << '\n';
    }

    // Clean up and exit
    free(grid);
    return 0;
}
