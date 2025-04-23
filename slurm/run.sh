#!/bin/bash
## SLURM Variables:
#SBATCH --job-name tm_tamp_tamix_tmp_tzw_32ngf_ncat
#SBATCH --output=/data/vision/polina/users/sebodiaz/projects/pose_fin/logs/pose.out
#SBATCH -e /data/vision/polina/users/sebodiaz/projects/pose/errors/%x-%j.err
#SBATCH -o /data/vision/polina/users/sebodiaz/projects/pose/outputs/%x-%j.out
#SBATCH --partition=polina-a6000
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6-00:00:00

# activate virtual environment
source /data/vision/polina/users/sebodiaz/miniconda3/bin/activate pose
export PYTHONPATH="/data/vision/polina/users/sebodiaz/projects/pose_fin:${PYTHONPATH}"
bash /data/vision/polina/users/sebodiaz/projects/pose_fin/configs/train-temporal.sh
