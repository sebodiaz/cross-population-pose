#!/bin/bash
## SLURM Variables:
#SBATCH --job-name b_small
#SBATCH --output=/data/vision/polina/users/sebodiaz/projects/pose/logs/pose.out
#SBATCH -e /data/vision/polina/users/sebodiaz/projects/pose/errors/%x-%j.err
#SBATCH -o /data/vision/polina/users/sebodiaz/projects/pose/outputs/%x-%j.out
#SBATCH --partition=polina-a6000
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=8-00:00:00

# activate virtual environment
source /data/vision/polina/users/sebodiaz/miniconda3/bin/activate pose
export PYTHONPATH="/data/vision/polina/users/sebodiaz/projects/pose:${PYTHONPATH}"
bash /data/vision/polina/users/sebodiaz/projects/pose/configs/train-base.sh
