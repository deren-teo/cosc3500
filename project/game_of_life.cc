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

// Given a byte representing a cell state and neighbour sum in the form:
//     <3 bits: unused><4 bits: neighbour sum><1 bit: state>
// kStateChanges maps the byte to 1 if the state will change, else 0
const int kStateChanges = 0x2AA4A; // bin: 101010101001001010

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
            grid[idx + j] = std::rand() & 0x01;
        }
    }
}

/**
 * Pre-calculates the neighbourhood sums of all cells before first evolution.
 * Assumes the grid is zero-padded on each side. Calculates these sums too,
 * to avoid underflow errors during evolution.
 *
 * @param grid Pointer to the memory allocated for the grid
*/
static void GridPrecalculate(char *grid) {
    // Corner indices of zero-padding
    // NOTE: top-left corner is just index 0
    int tr = row_size - 1;
    int bl = (n_cols + 1) * row_size;
    int br = (n_cols + 2) * row_size - 1;
    // Top left corner
    grid[0] = (grid[row_size + 1] & 0x01) << 1;
    // Top right corner
    grid[tr] = (grid[row_size + row_size - 2] & 0x01) << 1;
    // Botton left corner
    grid[bl] = (grid[n_cols * row_size + 1] & 0x01) << 1;
    // Botton right corner
    grid[br] = (grid[(n_cols + 1) * row_size - 2] & 0x01) << 1;
    // Top edge, except corners
    int sum = 0;
    for (int j = 1; j < tr; j++) {
        int j_blw = j + row_size;
        sum =  grid[j_blw - 1] & 0x01;
        sum += grid[j_blw] & 0x01;
        sum += grid[j_blw + 1] & 0x01;
        grid[j] |= sum << 1;
    }
    // Bottom edge, except corners
    for (int j = bl + 1; j < br; j++) {
        int j_abv = j - row_size;
        sum =  grid[j_abv - 1] & 0x01;
        sum += grid[j_abv] & 0x01;
        sum += grid[j_abv + 1] & 0x01;
        grid[j] |= sum << 1;
    }
    // Left edge, except corners
    for (int i = row_size; i < bl; i += row_size) {
        int i_rgt = i + 1;
        sum =  grid[i_rgt - row_size] & 0x01;
        sum += grid[i_rgt] & 0x01;
        sum += grid[i_rgt + row_size] & 0x01;
        grid[i] |= sum << 1;
    }
    // Right edge, except corners
    for (int i = tr + row_size; i < br; i += row_size) {
        int i_lft = i - 1;
        sum =  grid[i_lft - row_size] & 0x01;
        sum += grid[i_lft] & 0x01;
        sum += grid[i_lft + row_size] & 0x01;
        grid[i] |= sum << 1;
    }
    // Rest of the grid
    int idx = 0;
    for (int i = 0; i < n_rows; i++) {
        idx += row_size;
        for (int j = 1; j <= n_cols; j++) {
            int idx_j = idx + j;
            int idx_abv = idx_j - row_size;
            int idx_blw = idx_j + row_size;
            sum =  grid[idx_abv - 1] & 0x01;
            sum += grid[idx_abv] & 0x01;
            sum += grid[idx_abv + 1] & 0x01;
            sum += grid[idx_j - 1] & 0x01;
            // NOTE: a cell is not included in its own neighbour count
            sum += grid[idx_j + 1] & 0x01;
            sum += grid[idx_blw - 1] & 0x01;
            sum += grid[idx_blw] & 0x01;
            sum += grid[idx_blw + 1] & 0x01;
            grid[idx_j] |= sum << 1;
        }
    }
}

/**
 * Sets the cell with the given index to the dead state and decrements the
 * neighbour sum count of all surrounding cells.
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param idx Index of the cell whose state to change
*/
static void GridSetDead(char *grid, int idx) {
    int idx_abv = idx - row_size;
    int idx_blw = idx + row_size;
    grid[idx_abv - 1] -= 2;
    grid[idx_abv]     -= 2;
    grid[idx_abv + 1] -= 2;
    grid[idx - 1]     -= 2;
    grid[idx]         &= 0xFE;
    grid[idx + 1]     -= 2;
    grid[idx_blw - 1] -= 2;
    grid[idx_blw]     -= 2;
    grid[idx_blw + 1] -= 2;
}

static void GridSetLive(char *grid, int idx) {
    int idx_abv = idx - row_size;
    int idx_blw = idx + row_size;
    grid[idx_abv - 1] += 2;
    grid[idx_abv]     += 2;
    grid[idx_abv + 1] += 2;
    grid[idx - 1]     += 2;
    grid[idx]         |= 0x01;
    grid[idx + 1]     += 2;
    grid[idx_blw - 1] += 2;
    grid[idx_blw]     += 2;
    grid[idx_blw + 1] += 2;
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
    int idx = 0;
    for (int i = 0; i < n_rows; i++) {
        idx += row_size;
        for (int j = 1; j <= n_cols; j++) {
            int idxj = idx + j;
            if (kStateChanges & (1 << grid[idxj])) {
                if (grid[idxj] & 0x01) {
                    GridSetDead(evolved_grid, idxj);
                } else {
                    GridSetLive(evolved_grid, idxj);
                }
                *is_static = 0;
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
    GridPrecalculate(grid);
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
