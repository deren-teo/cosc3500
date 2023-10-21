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

python deserializer.py game_of_life.out 120 120 > game_of_life.txt
