/*******************************************************************************
 * @file    main.c
 * @author  Deren Teo
 * @date    11 August 2023 (last updated)
 * @brief   An optimised implementation of Conway's Game of Life.
 ******************************************************************************/

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

unsigned int N; /* Number of grid rows */
unsigned int M; /* Number of grid cols */

/**
 * Initialise a grid of size NxM elements, within which Life will evolve.
 *
 * @param N Number of rows
 * @param M Number of columns
 *
 * @return Pointer to the grid object.
*/
static inline char *grid_init(unsigned int N, unsigned int M) {

    /* Allocate just enough bytes to represent each cell by one bit */
    return malloc((N * M) / 8 + 1);
}

/**
 * Sets the alive (true) or dead (false) state of the grid cell at (n, m).
*/
static void grid_set_state(unsigned int n, unsigned int m, bool state) {

    return; /* TODO */
}

/**
 * Returns the sum of the 3x3 grid of cells centred at (n, m).
*/
static uint8_t grid_get_9sum(unsigned int n, unsigned int m) {

    return 0; /* TODO */
}

/**
 * Evolve the grid at the given pointer by applying the standard rules of Life.
 *
 * Rules in condensed format, looking at sum of 3x3 grid:
 *   1. If sum is 3, centre cell is lives
 *   2. If sum is 4, center cell retains current state
 *   3. If sum is anything else, center cell dies
 *
 * @param pGrid Pointer to the grid object
*/
static void grid_evolve(char *pGrid) {

    /* TODO: optimise */

    for (unsigned int n = 0; n < N; n++) {
        for (unsigned int m = 0; m < M; m++) {
            /* TODO: in case 3 and default, should I check if the state needs
                to change, or just write the value regardless? */
            switch (grid_get_9sum(n, m)) {
                case 3:
                    grid_set_state(n, m, true);
                    break;
                case 4:
                    break;
                default:
                    grid_set_state(n, m, false);
                    break;
            }
        }
    }
}

int main(int argc, char **argv) {

    /* Parse input arguments */
    if (argc < 4) {
        printf("Usage: %s N M i", argv[0]);
    }

    /* Create the grid */
    char *pGrid = grid_init(atoi(argv[1]), atoi(argv[2]));

    /* TODO: initial configuration */

    /* Evolve the simulation the specified number of iterations */
    for (uint8_t i = 0; i < atoi(argv[3]); i++) {
        grid_evolve(pGrid);
    }

    /* Clean up and exit */
    free(pGrid);

    return 0;
}
