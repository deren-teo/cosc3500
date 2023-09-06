/*******************************************************************************
 * @file    parser.h
 * @author  Deren Teo
 * @brief   A library for parsing run length encoded (RLE) files describing
 *          Game of Life patterns. Assumes file formatting as described by:
 *          https://conwaylife.com/wiki/Run_Length_Encoded
 ******************************************************************************/

#ifndef PARSER_H
#define PARSER_H

/**
 * Parse a run length encoded (RLE) pattern file and onto a Life grid.
 * Assumes the file is formatted as described by the LifeWiki article:
 *      https://conwaylife.com/wiki/Run_Length_Encoded
 *
 * If n_rows or n_cols is smaller than the size of the pattern, the size of the
 * grid is increased in that direction to be equal to the size of the pattern.
 * Otherwise, the pattern is centred in the grid.
 *
 * Creates a grid with a zero-padded border of width 1 cell. Hence, indexing
 * into the grid should happen between indices 1 and n_rows or n_cols.
 *
 * @param filename Name of the RLE file to parse
 * @param n_rows Pointer to variable to store number of rows in allocated grid
 * @param n_cols Pointer to variable to store number of cols in allocated grid
 *
 * @returns Pointer to the grid object initialised with the parsed pattern.
*/
char *ParseRLEFile(const char *filename, int *n_rows, int *n_cols);

/**
 * Greedily parses the given string as an integer until the first non-digit.
 *
 * @param buffer String to parse; must contain at least one digit
 * @param idx_start Index of the string to start parsing; must be an integer
 * @param idx_end Pointer to variable storing index of string after the last
 *          consecutive integer found
 *
 * @return Parsed integer containing as many consecutive digits as possible from
 * the given string, starting at the given starting index.
*/
int ParseIntGreedy(const char *buffer, int idx_start, int *idx_end);

#endif  // PARSER_H
