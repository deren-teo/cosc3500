#!/bin/bash
#SBATCH --job-name=s4528554-cosc
#SBATCH --partition=cosc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G # memory
#SBATCH --time=0-00:10 # time (D-HH:MM)

time ./game_of_life -i 1000 -s 2000x2000 -v
