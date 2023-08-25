/*******************************************************************************
 * @file    parser.h
 * @author  Deren Teo
 * @brief   A library for parsing run length encoded (RLE) files describing
 *          Game of Life patterns. Assumes file formatting as described by:
 *          https://conwaylife.com/wiki/Run_Length_Encoded
 ******************************************************************************/

#ifndef PARSER_H
#define PARSER_H

#include <cstdint>

/**
 * Parse a run length encoded (RLE) pattern file and onto a Life grid.
 * Assumes the file is formatted as described by the LifeWiki article:
 *      https://conwaylife.com/wiki/Run_Length_Encoded
 *
 * @param filename Name of the RLE file to parse
 * @param n_rows Pointer to variable to store number of rows in allocated grid
 * @param n_cols Pointer to variable to store number of cols in allocated grid
 *
 * @returns Pointer to the grid object initialised with the parsed pattern.
*/
char *ParseRLEFile(const char *filename, int *n_rows, int *n_cols);

#endif  // PARSER_H
