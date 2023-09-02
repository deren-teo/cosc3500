# COSC3500 Major Project: Cellular Automaton

The aim of this project is to create an optimised implementation of the well-known cellular automaton, Conway's Game of Life (abbr. Life). This project is implemented in two stages, differentiated by the types of optimsations applied.

**Stage 1** implements serial optimisation techniques, foremost of which are efficient memory access patterns.

**Stage 2** implements parallel optimisation techniques on top of the Stage 1 optimisations. These include AVX intrinsics, openMP, MPI and/or CUDA.

The most up-to-date working version of the project will always reside on the `main` branch. However, after the end of the Stage 1, a copy of the project in its serially-optimised state will be maintained on a separate branch.

This project mostly conforms with the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#C++_Version).

## Getting Started

### Installation

Clone the repository:

```bash
git clone git@github.com:deren-teo/cosc3500
```

## Usage
