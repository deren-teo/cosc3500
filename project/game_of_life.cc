/*******************************************************************************
 * @file    main.cc
 * @author  Deren Teo
 * @brief   An optimised implementation of Conway's Game of Life (abbr. Life).
 ******************************************************************************/

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "parser.h"

volatile int fp_argidx = 0; // argv idx of optional pattern file argument
volatile int n_iter = 99;   // maximum number of iterations to simulate
volatile int output = 0;    // whether to dump grid as binary to file
volatile int verbose = 0;   // whether to print runtime configuration to console

int n_rows = 0;             // placeholder "empty" number of rows in the grid
int n_cols = 0;             // placeholder "empty" number of columns in the grid

/**
 * Allocates memory for a 2D grid on which Life will evolve. Each cell is
 * allocated a single bit, since a cell has binary state.
 *
 * @param n_bytes Number of bytes allocated to the grid
 *
 * @return Pointer to the memory allocated for the grid.
*/
static char *GridCreateEmpty(const int n_bytes) {
    char *grid = static_cast<char *>(std::malloc(n_bytes));
    std::memset(grid, 0, n_bytes);
    return grid;
}

/**
 * Randomly initialise each cell in the grid as either alive (1) or dead (0).
 * If not manually seeded, std::rand() behaves as if seeded with std::srand(1).
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_bytes Number of bytes allocated to the grid
*/
static void GridRandomInit(char *grid, const int n_bytes) {
    for (int i = 0; i < n_bytes; i++) {
        grid[i] = static_cast<char>(std::rand() & 0xFF);
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
static int GridLocalSum(char *grid, const int row, const int col,
        const int n_rows, const int n_cols) {
    int cell_idx00 = (row - 1) * n_cols + (col - 1);
    int cell_idx10 = cell_idx00 + n_cols;
    int cell_idx20 = cell_idx10 + n_cols;
    int sum = 0;
    if (row > 0) {
        if (col > 0) {
            sum += !!(grid[cell_idx00 / 8] & (1 << (cell_idx00 % 8)));
        }
        sum += !!(grid[(cell_idx00 + 1) / 8] & (1 << ((cell_idx00 + 1) % 8)));
        if (col < (n_cols - 1)) {
            sum += !!(grid[(cell_idx00 + 2) / 8] & (1 << ((cell_idx00 + 2) % 8)));
        }
    }
    if (col > 0) {
        sum += !!(grid[cell_idx10 / 8] & (1 << (cell_idx10 % 8)));
    }
    sum += !!(grid[(cell_idx10 + 1) / 8] & (1 << ((cell_idx10 + 1) % 8)));
    if (col < (n_cols - 1)) {
        sum += !!(grid[(cell_idx10 + 2) / 8] & (1 << ((cell_idx10 + 2) % 8)));
    }
    if (row < (n_rows - 1)) {
        if (col > 0) {
            sum += !!(grid[cell_idx20 / 8] & (1 << (cell_idx20 % 8)));
        }
        sum += !!(grid[(cell_idx20 + 1) / 8] & (1 << ((cell_idx20 + 1) % 8)));
        if (col < (n_cols - 1)) {
            sum += !!(grid[(cell_idx20 + 2) / 8] & (1 << ((cell_idx20 + 2) % 8)));
        }
    }
    return sum;
}

/**
 * Returns the binary state of the cell (i, j).
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param row Row number of the cell
 * @param col Column number of the cell
 * @param n_cols Number of columns in the grid
 *
 * @return 1 if the cell is alive, else 0.
*/
static int CellGetState(char *grid, const int row, const int col,
        const int n_cols) {
    int cell_idx = row * n_cols + col;
    int byte_idx = cell_idx / 8;
    char byte_mask = 0 | (1 << (cell_idx % 8));
    return !!(grid[byte_idx] & byte_mask);
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
static void CellSetState(char *grid, const int row, const int col,
        const int n_cols, int state, int *state_changed) {
    int cell_idx = row * n_cols + col;
    int byte_idx = cell_idx / 8;
    char byte_mask = 0 | (1 << (cell_idx % 8));
    *state_changed = (!!(grid[byte_idx] & byte_mask) != state);
    if (state) {
        grid[byte_idx] |= byte_mask;
    } else {
        grid[byte_idx] &= ~byte_mask;
    }
}

/**
 * Updates the grid one timestep forward based on the standard rules of Life.
 * If no cell states changed, sets the early stopping flag to 1.
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_rows Number of rows in the grid
 * @param n_cols Number of columns in the grid
 * @param n_bytes Number of bytes allocated to the grid
 * @param stop_early Pointer to a variable to store early stopping flag
 *
 * @return Returns a pointer to the evolved grid.
*/
static char *GridEvolve(char *grid, const int n_rows, const int n_cols,
        const int n_bytes, int *stop_early) {
    // Allocate copy of grid to hold intermediate evolved cell states
    char *evolved_grid = static_cast<char *>(std::malloc(n_bytes));
    std::memcpy(evolved_grid, grid, n_bytes);
    // Early stopping condition: no cells changed (possibly all dead)
    *stop_early = 1;
    // Update cell states in grid copy
    int state_changed = 0;
    for (unsigned short i = 0; i < n_rows; i++) {
        for (unsigned short j = 0; j < n_cols; j++) {
            switch (GridLocalSum(grid, i, j, n_rows, n_cols)) {
                case 0: {
                    // All cells around (i, j) are dead; possibly early stop
                    break;
                }
                case 3: {
                    // Cell (i, j) now alive, regardless of previous state
                    if (!CellGetState(grid, i, j, n_cols)) {
                        CellSetState(evolved_grid, i, j, n_cols, 1, &state_changed);
                    }
                    *stop_early &= !state_changed;
                    break;
                }
                case 4: {
                    // (i, j) does not change state; possible early stop
                    break;
                }
                default: {
                    // Cell (i, j) now dead, regardless of previous state
                    if (CellGetState(grid, i, j, n_cols)) {
                        CellSetState(evolved_grid, i, j, n_cols, 0, &state_changed);
                    }
                    *stop_early &= !state_changed;
                    break;
                }
            }
        }
    }
    free(grid);
    return evolved_grid;
}

/**
 * Serialize grid state and write to file, preferably as fast as possible.
 *
 * @param fptr Pointer to the file object to write to
 * @param grid Pointer to the memory allocated for the grid
 * @param n_bytes Number of bytes allocated to the grid
*/
static inline void GridSerialize(std::FILE *fptr, char *grid, const int n_bytes) {
    std::fwrite(grid, 1, n_bytes, fptr);
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
        "  -f, --file FILEPATH  path to a pattern file in RLE format\n"
        "  -i, --iters NITER    number of iterations to simulate\n"
        "  -o, --output         output the grid in binary every iteration\n"
        "  -s, --size NxM       number of rows x columns in the grid \n"
        "  -v, --verbose        output runtime configuration to console\n\n";
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-f") || !std::strcmp(argv[i], "--file")) {
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
        } else if (!std::strcmp(argv[i], "-s") || !std::strcmp(argv[i], "--size")) {
            if (argc >= i + 1) {
                size_t buffer_size = std::strlen(argv[++i]);
                int start_idx = 0;
                int end_idx;
                n_rows = ParseIntGreedy(argv[i], buffer_size, start_idx, &end_idx);
                if (argv[i][end_idx] != 'x' || !std::isdigit(argv[i][end_idx + 1])) {
                    std::cout << help;
                    return 1;
                }
                start_idx = end_idx + 1;
                n_cols = ParseIntGreedy(argv[i], buffer_size, start_idx, &end_idx);
            } else {
                std::cout << help;
                return 1;
            }
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
    char *grid;
    int n_bytes;
    if (!n_rows) {
        n_rows = 100;   // default number of rows
    }
    if (!n_cols) {
        n_cols = 100;   // default number of columns
    }
    if (!fp_argidx) {
        n_bytes = std::ceil(n_rows * n_cols / 8.0);
        grid = GridCreateEmpty(n_bytes);
        GridRandomInit(grid, n_bytes);
    } else {
        grid = ParseRLEFile(argv[fp_argidx], &n_rows, &n_cols);
        n_bytes = std::ceil(n_rows * n_cols / 8.0);
    }
    // If verbose flag specified on the command line, print runtime config
    if (verbose) {
        std::cout << "Simulating Life on a " << n_rows << "x" << n_cols << " grid ";
        if (fp_argidx) {
            std::cout << "with initial pattern from \"" << argv[fp_argidx] << "\" ";
        } else {
            std::cout << "with a randomised initial pattern ";
        }
        std::cout << "for " << n_iter << " iterations... " << std::flush;
    }
    // Evolve the simulation the specified number of iterations or until
    // reaching a static state
    int stop_early;
    if (!output) {
        for (int i = 0; i < n_iter; i++) {
            grid = GridEvolve(grid, n_rows, n_cols, n_bytes, &stop_early);
            if (stop_early) {
                break;
            }
        }
    // If output flag specified on the command line, then every iteration the
    // grid is written as raw binary to "game_of_life.out"
    } else {
        std::FILE *fptr = std::fopen("game_of_life.out", "wb");
        for (int i = 0; i < n_iter; i++) {
            GridSerialize(fptr, grid, n_bytes);
            grid = GridEvolve(grid, n_rows, n_cols, n_bytes, &stop_early);
            if (stop_early) {
                break;
            }
        }
        GridSerialize(fptr, grid, n_bytes);
        std::fclose(fptr);
    }
    if (verbose) {
        std::cout << "(Done)\n";
    }
    // Clean up and exit
    free(grid);
    return 0;
}
