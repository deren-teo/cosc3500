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

volatile int fp_argidx = 0; // argv idx of optional pattern file argument
volatile int n_iter = 99;   // maximum number of iterations to simulate
volatile int output = 0;    // whether to dump grid as binary to file
volatile int verbose = 0;   // whether to print runtime configuration to console

/**
 * Allocates memory for a 2D grid on which Life will evolve.
 *
 * @param n_cells Number of cells in the grid; i.e. n_rows * n_cells
 *
 * @return Pointer to the memory allocated for the grid.
*/
static uint8_t *GridCreateEmpty(const int n_cells) {
    uint8_t *grid = static_cast<uint8_t *>(std::malloc(n_cells));
    std::memset(grid, 0, n_cells);
    return grid;
}

/**
 * Randomly initialise each cell in the grid as either alive (1) or dead (0).
 * If not manually seeded, std::rand() behaves as if seeded with std::srand(1).
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_cells Number of cells in the grid; i.e. n_rows * n_cells
*/
static void GridRandomInit(uint8_t *grid, const int n_cells) {
    for (int i = 0; i < n_cells; i++) {
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
        std::malloc(n_rows * n_cols));
    std::memcpy(evolved_grid, grid, n_rows * n_cols);

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
                    CellSetState(evolved_grid, i, j, n_cols, 1, &state_changed);
                    stop_early &= !state_changed;
                    break;
                }
                case 4: {
                    // (i, j) does not change state; possible early stop
                    break;
                }
                default: {
                    // Cell (i, j) now dead, regardless of previous state
                    CellSetState(evolved_grid, i, j, n_cols, 0, &state_changed);
                    stop_early &= !state_changed;
                    break;
                }
            }
        }
    }
    // Copy evolved cell states back into grid and free temporary grid copy
    // TODO: highly memory intensive; is there any way to point `grid` at
    //  `evolved_grid` and free the old grid?
    std::memcpy(grid, evolved_grid, n_rows * n_cols);
    free(evolved_grid);
    return stop_early;
}

/**
 * Serialize grid state and write to file, preferably as fast as possible.
 *
 * @param fptr Pointer to the file object to write to
 * @param grid Pointer to the memory allocated for the grid
 * @param n_cells Number of cells in the grid; i.e. n_rows * n_cols
*/
static inline void GridSerialize(std::FILE *fptr, uint8_t *grid, const int n_cells) {
    std::fwrite(grid, 1, n_cells, fptr);
    std::fputc('\n', fptr);
}

/**
 * Parses command line arguments as various configurable runtime options
 * by setting respective global variables.
 *
 * @return 0 if successful.
*/
static int ParseCmdline(int argc, char *argv[]) {
    const char *help = \
        "Usage: ./game_of_life [OPTION]\n\n"
        "  -f, --file=FILEPATH  path to a pattern file in RLE format\n"
        "  -i, --iters=NITER    number of iterations to simulate\n"
        "  -o, --output         output the grid in binary every iteration\n"
        "  -v, --verbose        output runtime configuration to console\n\n";
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-f") || !std::strcmp(argv[i], "--filepath")) {
            if (argc >= i + 1) {
                fp_argidx = ++i;
            } else {
                std::cout << help;
                return 1;
            }
        } else if (!std::strcmp(argv[i], "-i") || !std::strcmp(argv[i], "--iters")) {
            if (argc >= i + 1) {
                n_iter = std::atoi(argv[++i]);
            } else {
                std::cout << help;
                return 1;
            }
        } else if (!std::strcmp(argv[i], "-o") || !std::strcmp(argv[i], "--output")) {
            output = 1;
        } else if (!std::strcmp(argv[i], "-v") || !std::strcmp(argv[i], "--verbose")) {
            verbose = 1;
        } else {
            std::cout << help;
            return 1;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    // Parse command line arguments as various configurable runtime options
    if (ParseCmdline(argc, argv)) {
        return 1;
    }
    // Create and initialise the grid (latter either randomly or with a pattern)
    uint8_t *grid;
    int n_rows = 10; // default number of rows in grid
    int n_cols = 10; // default number of columns in grid
    int n_cells;     // i.e. n_rows * n_cols
    if (!fp_argidx) {
        n_cells = n_rows * n_cols;
        grid = GridCreateEmpty(n_cells);
        GridRandomInit(grid, n_cells);
    } else {
        grid = ParseRLEFile(argv[fp_argidx], &n_rows, &n_cols);
        n_cells = n_rows * n_cols;
    }
    // If verbose flag specified on the command line, print runtime config
    if (verbose) {
        std::cout << "Simulating Life on a " << n_rows << "x" << n_cols << " grid ";
        if (fp_argidx) {
            std::cout << "with initial pattern from \"" << argv[fp_argidx] << "\" ";
        } else {
            std::cout << "with a randomised initial pattern ";
        }
        std::cout << "for " << n_iter << " iterations... ";
    }
    // Evolve the simulation the specified number of iterations or until all
    // cells are dead (meaning nothing will happen in all future iterations)
    if (!output) {
        for (int i = 0; i < n_iter; i++) {
            if (GridEvolve(grid, n_rows, n_cols) == 1) {
                break;
            }
        }
    // If output flag specified on the command line, then every iteration the
    // grid is written as raw binary to "game_of_life.out"
    } else {
        std::FILE *fptr = std::fopen("game_of_life.out", "wb");
        for (int i = 0; i < n_iter; i++) {
            GridSerialize(fptr, grid, n_cells);
            if (GridEvolve(grid, n_rows, n_cols) == 1) {
                break;
            }
        }
        GridSerialize(fptr, grid, n_cells);
        std::fclose(fptr);
    }
    if (verbose) {
        std::cout << "(Done)\n";
    }
    // Clean up and exit
    free(grid);
    return 0;
}
