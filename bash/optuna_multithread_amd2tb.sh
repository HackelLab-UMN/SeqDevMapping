#!/bin/bash -l        
#SBATCH --time=12:00:00
#SBATCH -p amd2tb
#SBATCH --mail-type=ALL  

module load conda

conda activate Greg2024

python run.py $1
