#include "gridEvolve.h"

// Given a byte representing a cell state and neighbour sum in the form:
//     <3 bits: unused><4 bits: neighbour sum><1 bit: state>
// this value maps the byte to 1 if the state will change, else 0
#define TRANSITION_MAP 0x2AA4A  // bin: 101010101001001010

/**
 * @brief Kills the cell with the given index
 *
 * @param[in] grid Pointer to the memory allocated for the grid
 * @param[in] idx Index of the cell whose state to change
 * @param[in] rowSize Number of columns plus zero-padding
 * @return void
*/
static void gridSetDead(char *grid, const int idx, const int rowSize) {
    int idx_abv = idx - rowSize;
    int idx_blw = idx + rowSize;
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

/**
 * @brief Revives the cell with the given index
 *
 * @param[in] grid Pointer to the memory allocated for the grid
 * @param[in] idx Index of the cell whose state to change
 * @param[in] rowSize Number of columns plus zero-padding
 * @return void
*/
static void gridSetLive(char *grid, const int idx, const int rowSize) {
    int idx_abv = idx - rowSize;
    int idx_blw = idx + rowSize;
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

void gridEvolve(char *grid, const int nRows, const int nCols, int *isStatic) {
    // Number of columns plus zero-padding
    const int rowSize = nCols + 2;
    // Determine which cell states change in the next generation
    std::vector<int> changedIdxs;
    int idx = 0;
    for (int i = 0; i < nRows; i++) {
        idx += rowSize;
        for (int j = 1; j < nCols; j++) {
            int idxj = idx + j;
            if (TRANSITION_MAP & (1 << grid[idxj])) {
                changedIdxs.push_back(idxj);
            }
        }
    }
    // If no cell states changed, set is_static to True and return early
    if (changedIdxs.empty()) {
        *isStatic = 1;
        return;
    }
    // Otherwise, go through and update cells which should change
    const int vEnd = changedIdxs.size();
    for (int i = 0; i < vEnd; i++) {
        idx = changedIdxs[i];
        if (grid[idx] & 0x01) {
            gridSetDead(grid, idx, rowSize);
        } else {
            gridSetLive(grid, idx, rowSize);
        }
    }
}
