#!/bin/bash
##SBATCH -N 2
##SBATCH --gres=gpu:8
#SBATCH --gres=gpu:1
##SBATCH --ntasks-per-node=6
##SBATCH --cpus-per-task=1
##SBATCH --qos=gpugpu
module load anaconda
source activate overlap
export PYTHONUNBUFFERED=1 
srun python main.py 
