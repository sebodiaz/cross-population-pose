#!/bin/bash
## SLURM Variables:
#SBATCH --job-name lab
#SBATCH --output=/data/vision/polina/users/sebodiaz/projects/pose_fin/logs/pose.out
#SBATCH -e /data/vision/polina/users/sebodiaz/projects/pose_fin/errors/%x-%j.err
#SBATCH -o /data/vision/polina/users/sebodiaz/projects/pose_fin/outputs/%x-%j.out
#SBATCH --partition=polina-a6000
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=5-00:00:00

# activate virtual environment
source /data/vision/polina/users/sebodiaz/miniconda3/bin/activate pose
export PYTHONPATH="/data/vision/polina/users/sebodiaz/projects/pose:${PYTHONPATH}"
bash /data/vision/polina/users/sebodiaz/projects/pose_fin/configs/finetune_inference_labeling.sh
