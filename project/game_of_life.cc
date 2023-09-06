/*******************************************************************************
 * @file    main.cc
 * @author  Deren Teo
 * @brief   An optimised implementation of Conway's Game of Life (abbr. Life).
 ******************************************************************************/

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "parser.h"

volatile int fp_argidx = 0;  // argv idx of optional pattern file argument
volatile int n_iter    = 99; // maximum number of iterations to simulate
volatile int output    = 0;  // flag to dump grid as binary to file
volatile int verbose   = 0;  // flag to print runtime configuration to console
volatile int row_size  = 0;  // zero-padded row size; i.e. n_cols + 2

int n_rows = 0; // number of rows in the grid, not including zero-padding
int n_cols = 0; // number of columns in the grid, not including zero-padding

/**
 * Allocates memory for a 2D grid on which Life will evolve. Each cell is
 * allocated a single bit, since a cell has binary state.
 *
 * @param n_bytes Number of bytes allocated to the grid, including zero-padding
 *
 * @return Pointer to the memory allocated for the grid.
*/
static char *GridCreateEmpty(const int n_bytes) {
    char *grid = static_cast<char *>(std::malloc(n_bytes));
    std::memset(grid, 0, n_bytes);
    return grid;
}

/**
 * Randomly initialise each cell in the grid as either alive (1) or dead (0). If
 * not seeded during command line parsing, std::rand() behaves as if seeded with
 * std::srand(1). Assumes the grid has one cell of zero-padding all around the
 * border, and does not write to these cells.
 *
 * @param grid Pointer to the memory allocated for the grid
*/
static void GridRandomInit(char *grid) {
    int idx = 0;
    for (int i = 0; i < n_rows; i++) {
        idx += row_size;
        for (int j = 1; j <= n_cols; j++) {
            grid[idx + j] = static_cast<char>(std::rand());
        }
    }
}

/**
 * Returns the sum of the 3x3 grid of cells around and including (row, col).
 * Assumes the grid is zero-padded around the border, hence no special
 * consideration is taken with cells otherwise at the edge of the grid.
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param row Row number of centre cell, indexed from 1 to n_rows
 * @param col Column number of centre cell, indexed from 1 to n_rows
 *
 * @return Sum of the 3x3 grid of cells centred on (row, col).
*/
static int GridLocalSum(char *grid, const int row, const int col) {
    int idx_ctr = row * row_size + col;
    int idx_abv = idx_ctr - row_size;
    int idx_blw = idx_ctr + row_size;

    int sum = 0;
    sum += grid[idx_abv - 1];
    sum += grid[idx_abv];
    sum += grid[idx_abv + 1];
    sum += grid[idx_ctr - 1];
    sum += grid[idx_ctr];
    sum += grid[idx_ctr + 1];
    sum += grid[idx_blw - 1];
    sum += grid[idx_blw];
    sum += grid[idx_blw + 1];
    return sum;
}

/**
 * Updates the grid one timestep forward based on the standard rules of Life.
 * If no cell states changed this iteration, sets the `is_static` flag to 1.
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_bytes Number of bytes allocated to the grid, including zero-padding
 * @param is_static Pointer to an external flag indicating whether grid has
 *      reached a static state or is still evolving
 *
 * @return Returns a pointer to the evolved grid.
*/
static char *GridEvolve(char *grid, const int n_bytes, int *is_static) {
    // Allocate copy of grid to write evolved cell states
    char *evolved_grid = static_cast<char *>(std::malloc(n_bytes));
    std::memcpy(evolved_grid, grid, n_bytes);

    // Update cell states in grid copy
    *is_static = 1;
    for (int i = 1; i <= n_rows; i++) {
        for (int j = 1; j <= n_cols; j++) {
            switch (GridLocalSum(grid, i, j)) {
                case 0: {
                    // All cells around (i, j) are dead; possibly early stop
                    break;
                }
                case 3: {
                    // Cell (i, j) now alive, regardless of previous state
                    if (grid[i * row_size + j] == 0) {
                        evolved_grid[i * row_size + j] = 1;
                    }
                    *is_static = 0;
                    break;
                }
                case 4: {
                    // (i, j) does not change state; possible early stop
                    break;
                }
                default: {
                    // Cell (i, j) now dead, regardless of previous state
                    if (grid[i * row_size + j] == 1) {
                        evolved_grid[i * row_size + j] = 0;
                    }
                    *is_static = 0;
                    break;
                }
            }
        }
    }
    free(grid);
    return evolved_grid;
}

/**
 * Dumps the given grid to the given file pointer, not including zero-padding.
 *
 * @param fptr Pointer to the file object to write to
 * @param grid Pointer to the memory allocated for the grid
*/
static void GridSerialize(std::FILE *fptr, char *grid) {
    int idx = 0;
    for (int i = 0; i < n_rows; i++) {
        idx += row_size;
        std::fwrite(grid + idx + 1, 1, n_cols, fptr);
    }
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
        "      --seed SEED      random seed (default: 1)\n"
        "  -s, --size NxM       number of rows x columns in the grid\n"
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
        } else if (!std::strcmp(argv[i], "--seed")) {
            if (argc >= i + 1) {
                std::srand(std::atoi(argv[++i]));
            } else {
                std::cout << help;
                return 1;
            }
        } else if (!std::strcmp(argv[i], "-s") || !std::strcmp(argv[i], "--size")) {
            if (argc >= i + 1) {
                int start_idx = 0;
                int end_idx;
                n_rows = ParseIntGreedy(argv[++i], start_idx, &end_idx);
                if (argv[i][end_idx] != 'x' || !std::isdigit(argv[i][end_idx + 1])) {
                    std::cout << help;
                    return 1;
                }
                start_idx = end_idx + 1;
                n_cols = ParseIntGreedy(argv[i], start_idx, &end_idx);
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
    int n_bytes; // number of bytes allocated to the grid, including zero-padding
    if (!n_rows) {
        n_rows = 100;   // default number of rows
    }
    if (!n_cols) {
        n_cols = 100;   // default number of columns
    }
    if (!fp_argidx) {
        row_size = n_cols + 2;
        n_bytes = (n_rows + 2) * row_size;
        grid = GridCreateEmpty(n_bytes);
        GridRandomInit(grid);
    } else {
        grid = ParseRLEFile(argv[fp_argidx], &n_rows, &n_cols);
        row_size = n_cols + 2;
        n_bytes = (n_rows + 2) * row_size;
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
    int is_static = 0;
    if (!output) {
        for (int i = 0; i < n_iter; i++) {
            grid = GridEvolve(grid, n_bytes, &is_static);
            if (is_static) {
                break;
            }
        }
    // If output flag specified on the command line, then every iteration the
    // grid is written as raw binary to "game_of_life.out"
    } else {
        std::FILE *fptr = std::fopen("game_of_life.out", "wb");
        for (int i = 0; i < n_iter; i++) {
            GridSerialize(fptr, grid);
            grid = GridEvolve(grid, n_bytes, &is_static);
            if (is_static) {
                break;
            }
        }
        GridSerialize(fptr, grid);
        std::fclose(fptr);
    }
    if (verbose) {
        std::cout << "(Done)\n";
    }
    // Clean up and exit
    free(grid);
    return 0;
}
