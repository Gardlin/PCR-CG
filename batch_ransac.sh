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
source_path=/HOME/scz0430/run/train_LoMatch_superglue_window_size5/3DLoMatch/1_gpu_2_img_initmode_pri3d_256_gamma0.95_lr0.005_finalfeatsdim_64/test/pth
res_path=/HOME/scz0430/run/train_LoMatch_superglue_window_size5/3DLoMatch/1_gpu_2_img_initmode_pri3d_256_gamma0.95_lr0.005_finalfeatsdim_64/test/est_traj
dataset=3DLoMatch
gt_dir=./configs/benchmarks/$dataset
for N_POINTS in 250  500 1000 2500 5000
do
 srun  python scripts/evaluate_predator.py --source_path $source_path --benchmark $dataset --exp_dir $res_path --gt_folder $gt_dir  --n_points $N_POINTS
done
