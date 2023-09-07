#!/bin/bash
#SBATCH --job-name=s4528554-benchmark-life
#SBATCH --partition=cosc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G # memory
#SBATCH --time=0-00:10 # time (D-HH:MM)
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

make clean
make

# Grid size: 10x10
time ./game_of_life -i 1000 --seed 1 -s 10x10 -v
time ./game_of_life -i 1000 --seed 2 -s 10x10 -v
time ./game_of_life -i 1000 --seed 42 -s 10x10 -v

# Grid size: 100x100
time ./game_of_life -i 1000 --seed 1 -s 100x100 -v
time ./game_of_life -i 1000 --seed 2 -s 100x100 -v
time ./game_of_life -i 1000 --seed 42 -s 100x100 -v

# Grid size: 1000x1000
time ./game_of_life -i 1000 --seed 1 -s 1000x1000 -v
time ./game_of_life -i 1000 --seed 2 -s 1000x1000 -v
time ./game_of_life -i 1000 --seed 42 -s 1000x1000 -v
