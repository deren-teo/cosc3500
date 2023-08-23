/*******************************************************************************
 * @file    parser.cc
 * @author  Deren Teo
 * @brief   A library for parsing run length encoded (RLE) files describing
 *          Game of Life patterns. Assumes file formatting as described by:
 *          https://conwaylife.com/wiki/Run_Length_Encoded
 ******************************************************************************/

#include "parser.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

/**
 * Greedily parses the string as an integer until the first non-digit.
*/
static int ParseIntGreedy(const char *buffer, int buffer_size, int idx_start,
        int *idx_end);

/**
 * Parses the number of rows and columns into the given variables.
*/
static void ParseRLEGridSize(const char *buffer, int buffer_size, int *n_rows,
        int *n_cols);

/**
 * Parses a pattern line and adds the pattern to the grid starting at the given
 * row and column indices.
*/
static void ParseRLEPatternLine(const char *buffer, int buffer_size,
        uint8_t *grid, const int n_cols, int *row_idx, int *col_idx);

/**
 * Adds `run_count` number of live cells to the grid, starting at the given
 * row and column.
*/
static void GridAddPattern(uint8_t *grid, int n_cols, int row_idx, int col_idx,
        int run_count);

/** EXTERNAL FUNCTION DEFINITIONS *********************************************/

uint8_t *ParseRLEFile(const char *filename, int *n_rows, int *n_cols) {
    std::FILE *fptr = std::fopen(filename, "r");

    // LifeWiki RLE format description specifies lines must not exceed 70 chars,
    // but newline, carriage return and null characters add to this
    const int kBufferSize = 80;
    char buffer[kBufferSize] = {0};

    uint8_t *grid;
    int row_idx = 0;
    int col_idx = 0;

    fgets(buffer, kBufferSize, fptr);
    while (!std::feof(fptr)) {
        switch (buffer[0]) {
            // Line starting with '#' is a comment, which we ignore
            case '#': {
                break;
            }
            // Line starting with 'x' gives grid dimensions
            case 'x': {
                ParseRLEGridSize(buffer, kBufferSize, n_rows, n_cols);
                grid = static_cast<uint8_t *>(std::malloc((*n_rows) * (*n_cols)));
                break;
            }
            // Line starting with anything else is parsed as a pattern line
            default: {
                ParseRLEPatternLine(buffer, kBufferSize, grid, *n_cols,
                    &row_idx, &col_idx);
                break;
            }
        }
        fgets(buffer, kBufferSize, fptr);
    }
    std::fclose(fptr);
    return grid;
}

/** INTERNAL FUNCTION DEFINITIONS *********************************************/

static int ParseIntGreedy(const char *buffer, int buffer_size, int idx_start,
        int *idx_end) {

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

static void ParseRLEGridSize(const char *buffer, int buffer_size, int *n_rows,
        int *n_cols) {

    int idx_start = 0;
    int idx_end;

    // Parse the "x" dimension by looking for the first string of numbers
    while (!std::isdigit(buffer[idx_start++]));
    *n_cols = ParseIntGreedy(buffer, buffer_size, idx_start - 1, &idx_end);

    // Parse the "y" dimension by looking for the next string of numbers
    idx_start = idx_end + 1;
    while (!std::isdigit(buffer[idx_start++]));
    *n_rows = ParseIntGreedy(buffer, buffer_size, idx_start - 1, &idx_end);
}

static void ParseRLEPatternLine(const char *buffer, int buffer_size,
        uint8_t *grid, const int n_cols, int *row_idx, int *col_idx) {

    int idx_end;
    int run_count = 1;

    for (int i = 0; i < buffer_size; i++) {
        if (buffer[i] == '\0' || buffer[i] == '!') {
            return;
        }
        if (buffer[i] == 'b' ) {
            *col_idx += run_count;
            run_count = 1;
            continue;
        }
        if (buffer[i] == 'o') {
            GridAddPattern(grid, n_cols, *row_idx, *col_idx, run_count);
            *col_idx += run_count;
            run_count = 1;
            continue;
        }
        if (buffer[i] == '$') {
            *row_idx += run_count;
            *col_idx = 0;
            run_count = 1;
            continue;
        }
        if (std::isdigit(buffer[i])) {
            run_count = ParseIntGreedy(buffer, buffer_size, i, &idx_end);
            i = idx_end - 1;
            continue;
        }
    }
}

static void GridAddPattern(uint8_t *grid, const int n_cols, const int row_idx,
        int col_idx, int run_count) {

    std::memset(grid + n_cols * row_idx + col_idx, 1, run_count);
}
