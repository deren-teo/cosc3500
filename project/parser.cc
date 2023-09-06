/*******************************************************************************
 * @file    parser.cc
 * @author  Deren Teo
 * @brief   A library for parsing run length encoded (RLE) files describing
 *          Game of Life patterns. Assumes file formatting as described by:
 *          https://conwaylife.com/wiki/Run_Length_Encoded
 ******************************************************************************/

#include "parser.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cstdio>

/**
 * Parses the size of the pattern in an RLE file format into a number of rows
 * and columns and writes these to the given pointers.
 *
 * @param buffer String representing a line of an RLE file
 * @param p_rows Pointer to the number of rows in the pattern
 * @param p_cols Pointer to the number of columns in the pattern
*/
static void ParseRLEPatternSize(const char *buffer, int *p_rows, int *p_cols);

/**
 * Parses a pattern line and adds the pattern to the grid starting at the given
 * row and column indices. Assumes the grid is zero-padded around the border.
 *
 * @param grid Pointer to memory allocated for the grid
 * @param buffer String representing a line of an RLE file
 * @param buffer_size Maximum length of buffer
 * @param n_cols Number of columns in the grid, not including zero-padding
 * @param t_pad Top padding between pattern and grid, including zero-padding
 * @param l_pad Left padding between pattern and grid, including zero-padding
 * @param p_row_idx Pointer to current row index into pattern
 * @param p_col_idx Pointer to current column index into pattern
*/
static void ParseRLEPatternLine(char *grid, const char *buffer,
        const int buffer_size, const int n_cols, const int t_pad,
        const int l_pad, int *p_row_idx, int *p_col_idx);

/**
 * Adds `run_count` number of live cells to the grid, starting at the given
 * row and column. Assumes the grid is zero-padded around the border.
 *
 * @param grid Pointer to memory allocated for the grid
 * @param n_cols Number of columns in the grid, not including zero-padding
 * @param row_idx Current row index into grid
 * @param col_idx Current column index into grid
 * @param run_count Number of consecutive "1"s to write at current grid position
*/
static void GridAddPattern(char *grid, int n_cols, int row_idx, int col_idx,
        int run_count);

/** EXTERNAL FUNCTION DEFINITIONS *********************************************/

char *ParseRLEFile(const char *filename, int *n_rows, int *n_cols) {
    std::FILE *fptr = std::fopen(filename, "r");

    // LifeWiki RLE format description specifies lines must not exceed 70 chars,
    // but newline, carriage return and null characters add to this
    const int kBufferSize = 80;
    char buffer[kBufferSize] = {0};

    char *grid = nullptr;
    int p_rows;         // number of rows in the pattern
    int p_cols;         // number of columns in the pattern
    int p_row_idx = 0;  // current row index into pattern (subset of grid)
    int p_col_idx = 0;  // current column index into pattern (subset of grid)
    int t_pad = 0;      // padding between top of pattern and top of grid
    int l_pad = 0;      // padding between left side of pattern and of grid

    while (!std::feof(fptr)) {
        fgets(buffer, kBufferSize, fptr);
        switch (buffer[0]) {
            // Line starting with '#' is a comment, which we ignore
            case '#': {
                break;
            }
            // Line starting with 'x' gives grid dimensions
            case 'x': {
                ParseRLEPatternSize(buffer, &p_rows, &p_cols);
                if (*n_rows < p_rows) {
                    *n_rows = p_rows;
                } else {
                    t_pad = (*n_rows - p_rows) / 2 + 1; // incl. zero-padding
                }
                if (*n_cols < p_cols) {
                    *n_cols = p_cols;
                } else {
                    l_pad = (*n_cols - p_cols) / 2 + 1; // incl. zero-padding
                }
                size_t n_bytes = (*n_rows + 2) * (*n_cols + 2);
                grid = static_cast<char *>(std::malloc(n_bytes));
                std::memset(grid, 0, n_bytes);
                break;
            }
            // Line starting with anything else is parsed as a pattern line
            default: {
                ParseRLEPatternLine(grid, buffer, kBufferSize, *n_cols,
                    t_pad, l_pad, &p_row_idx, &p_col_idx);
                break;
            }
        }
    }
    std::fclose(fptr);
    return grid;
}

int ParseIntGreedy(const char *buffer, int idx_start, int *idx_end) {

    // Greedily parse digits until the first non-digit
    *idx_end = idx_start + 1;
    while (std::isdigit(buffer[*idx_end])) {
        (*idx_end)++;
    }

    // Copy the digits as a substring, with an extra byte for null terminator
    int num_digits = *idx_end - idx_start;
    char *numstr = static_cast<char *>(std::malloc(num_digits + 1));
    std::memcpy(numstr, buffer + idx_start, num_digits);
    numstr[num_digits] = '\0';

    // Parse the substring and return the result
    int res = std::atoi(numstr);
    std::free(numstr);
    return res;
}

/** INTERNAL FUNCTION DEFINITIONS *********************************************/

static void ParseRLEPatternSize(const char *buffer, int *p_rows, int *p_cols) {
    int idx_start = 0;
    int idx_end;

    // Parse the "x" dimension by looking for the first string of numbers
    while (!std::isdigit(buffer[idx_start++]));
    *p_cols = ParseIntGreedy(buffer, idx_start - 1, &idx_end);

    // Parse the "y" dimension by looking for the next string of numbers
    idx_start = idx_end + 1;
    while (!std::isdigit(buffer[idx_start++]));
    *p_rows = ParseIntGreedy(buffer, idx_start - 1, &idx_end);
}

static void ParseRLEPatternLine(char *grid, const char *buffer,
        const int buffer_size, const int n_cols, const int t_pad,
        const int l_pad, int *p_row_idx, int *p_col_idx) {
    int idx_end;
    int run_count = 1;

    for (int i = 0; i < buffer_size; i++) {
        if (buffer[i] == '\0' || buffer[i] == '!') {
            return;
        }
        if (buffer[i] == 'b' ) {
            *p_col_idx += run_count;
            run_count = 1;
            continue;
        }
        if (buffer[i] == 'o') {
            GridAddPattern(grid, n_cols, *p_row_idx + t_pad, *p_col_idx + l_pad,
                run_count);
            *p_col_idx += run_count;
            run_count = 1;
            continue;
        }
        if (buffer[i] == '$') {
            *p_row_idx += run_count;
            *p_col_idx = 0;
            run_count = 1;
            continue;
        }
        if (std::isdigit(buffer[i])) {
            run_count = ParseIntGreedy(buffer, i, &idx_end);
            i = idx_end - 1;
            continue;
        }
    }
}

static inline void GridAddPattern(char *grid, const int n_cols, const int row_idx,
        int col_idx, int run_count) {
    std::memset(grid + (n_cols + 2) * row_idx + col_idx, 1, run_count);
}
