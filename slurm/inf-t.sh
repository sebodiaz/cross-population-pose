#!/bin/bash
## SLURM Variables:
#SBATCH --job-name test_set
#SBATCH --output=/data/vision/polina/users/sebodiaz/projects/pose_fin/logs/pose.out
#SBATCH -e /data/vision/polina/users/sebodiaz/projects/pose/errors/%x-%j.err
#SBATCH -o /data/vision/polina/users/sebodiaz/projects/pose/outputs/%x-%j.out
#SBATCH --partition=polina-2080ti
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=6-00:00:00

# activate virtual environment
source /data/vision/polina/users/sebodiaz/miniconda3/bin/activate pose
export PYTHONPATH="/data/vision/polina/users/sebodiaz/projects/pose_fin:${PYTHONPATH}"
bash /data/vision/polina/users/sebodiaz/projects/pose_fin/configs/inf_miccai.sh
