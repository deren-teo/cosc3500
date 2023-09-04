/*******************************************************************************
 * @file    main.cc
 * @author  Deren Teo
 * @brief   An optimised implementation of Conway's Game of Life (abbr. Life).
 ******************************************************************************/

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "parser.h"

volatile int fp_argidx = 0; // argv idx of optional pattern file argument
volatile int n_iter = 99;   // maximum number of iterations to simulate
volatile int output = 0;    // flag to dump grid as binary to file
volatile int verbose = 0;   // flag to print runtime configuration to console

int n_rows = 0;     // placeholder "empty" number of rows in the grid
int n_cols = 0;     // placeholder "empty" number of columns in the grid

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
static char *GridCreateEmpty(const int n_cells) {
    char *grid = static_cast<char *>(std::malloc(n_cells));
    std::memset(grid, 0, n_cells);
    return grid;
}

/**
 * Randomly initialise each cell in the grid as either alive (1) or dead (0).
 * If not manually seeded, std::rand() behaves as if seeded with std::srand(1).
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_bytes Number of bytes allocated to the grid
*/
static void GridRandomInit(char *grid, const int n_cells) {
    for (int i = 0; i < n_cells; i++) {
        grid[i] = static_cast<char>(std::rand() & 0x01);
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
            sum += grid[cell_idx00];
        }
        sum += grid[cell_idx00 + 1];
        if (col < (n_cols - 1)) {
            sum += grid[cell_idx00 + 2];
        }
    }
    if (col > 0) {
        sum += grid[cell_idx10];
    }
    sum += grid[cell_idx10 + 1];
    if (col < (n_cols - 1)) {
        sum += grid[cell_idx10 + 2];
    }
    if (row < (n_rows - 1)) {
        if (col > 0) {
            sum += grid[cell_idx20];
        }
        sum += grid[cell_idx20 + 1];
        if (col < (n_cols - 1)) {
            sum += grid[cell_idx20 + 2];
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
static inline int CellGetState(char *grid, const int row, const int col,
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
static inline void CellSetState(char *grid, const int row, const int col,
        const int n_cols, char state) {
    grid[row * n_cols + col] = state;
}

/**
 * Updates the grid one timestep forward based on the standard rules of Life.
 * If no cell states changed, sets the early stopping flag to 1.
 *
 * @param grid Pointer to the memory allocated for the grid
 * @param n_rows Number of rows in the grid
 * @param n_cols Number of columns in the grid
 * @param n_cells Number of cells in the grid
 * @param stop_early Pointer to a variable to store early stopping flag
 *
 * @return Returns a pointer to the evolved grid.
*/
static char *GridEvolve(char *grid, const int n_rows, const int n_cols,
        const int n_cells, int *stop_early) {
    // Allocate copy of grid to hold intermediate evolved cell states
    char *evolved_grid = static_cast<char *>(std::malloc(n_cells));
    std::memcpy(evolved_grid, grid, n_cells);

    *stop_early = 1;
    // Update cell states in grid copy
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
                        CellSetState(evolved_grid, i, j, n_cols, 1);
                    }
                    *stop_early = 0;
                    break;
                }
                case 4: {
                    // (i, j) does not change state; possible early stop
                    break;
                }
                default: {
                    // Cell (i, j) now dead, regardless of previous state
                    if (CellGetState(grid, i, j, n_cols)) {
                        CellSetState(evolved_grid, i, j, n_cols, 0);
                    }
                    *stop_early = 0;
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
 * @param n_cells Number of cells in the grid
*/
static inline void GridSerialize(std::FILE *fptr, char *grid, const int n_cells) {
    std::fwrite(grid, 1, n_cells, fptr);
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
    int n_cells;
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
            grid = GridEvolve(grid, n_rows, n_cols, n_cells, &stop_early);
            if (stop_early) {
                break;
            }
        }
    // If output flag specified on the command line, then every iteration the
    // grid is written as raw binary to "game_of_life.out"
    } else {
        std::FILE *fptr = std::fopen("game_of_life.out", "wb");
        for (int i = 0; i < n_iter; i++) {
            GridSerialize(fptr, grid, n_cells);
            grid = GridEvolve(grid, n_rows, n_cols, n_cells, &stop_early);
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
