#!/bin/bash

#SBATCH -n 2
#SBATCH -t 100:00:00
#SBATCH --mem=32G

#SBATCH -J dsamNMR
#SBATCH -o dsamNMR-%j.out
#SBATCH -e dsamNMR-%j.out

#call to script
python -u multi_dimension.py