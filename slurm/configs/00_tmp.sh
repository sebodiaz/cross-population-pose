#!/bin/bash
## SLURM Variables:
#SBATCH --job-name process
#SBATCH --output=/data/vision/polina/users/sebodiaz/projects/pose_fin/slurm/logs/pose.out
#SBATCH -e /data/vision/polina/users/sebodiaz/projects/pose_fin/slurm/errors/%x-%j.err
#SBATCH -o /data/vision/polina/users/sebodiaz/projects/pose_fin/slurm/outputs/%x-%j.out
#SBATCH --partition=polina-2080ti
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=15G
#SBATCH --time=6-00:00:00

# activate virtual environment
source /data/vision/polina/users/sebodiaz/miniconda3/bin/activate pose
export PYTHONPATH="/data/vision/polina/users/sebodiaz/projects/pose_fin:${PYTHONPATH}"
python /data/vision/polina/users/sebodiaz/projects/pose_fin/process-to-seg-labels.py
python /data/vision/polina/users/sebodiaz/projects/pose_fin/add-body-seg-to-keypoint.py