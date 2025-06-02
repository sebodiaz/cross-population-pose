#!/bin/bash
## SLURM Variables:
#SBATCH --job-name KP_AMIX
#SBATCH --output=/data/vision/polina/users/sebodiaz/projects/pose_fin/slurm/logs/pose.out
#SBATCH -e /data/vision/polina/users/sebodiaz/projects/pose_fin/slurm/errors/%x-%j.err
#SBATCH -o /data/vision/polina/users/sebodiaz/projects/pose_fin/slurm/outputs/%x-%j.out
#SBATCH --partition=polina-a6000
#SBATCH --qos=vision-polina-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6-00:00:00

# activate virtual environment
source /data/vision/polina/users/sebodiaz/miniconda3/bin/activate pose
export PYTHONPATH="/data/vision/polina/users/sebodiaz/projects/pose_fin:${PYTHONPATH}"
srun python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py \
  --stage train \
  --run_name E3_POSE_AMIX_CD50 \
  --save_path /data/vision/polina/users/sebodiaz/projects/pose_fin/runs/ \
  --seed 54 \
  --use_amp True \
  --use_amix True \
  --dropout 0.5 \
  --use_fabric True \
  --logger True \
  --num_nodes 1 \
  --num_gpus 1 \
  --model_name small_unet \
  --crop_size 96 \
  --custom_augmentation True \
  --rot True \
  --batch_size 8 \
  --seg_coeff 1.0 \
  --reg_coeff 20.0 \
  --learn_coeff True \
  --depth 4 \
  --nFeat 16 \
  --optimizer adamw \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --lr_scheduler linear \
  --epochs 500 \
  --val_freq 5 \
  --augmentation_prob 0.9 \
  --zoom 0.5 \
  --noise True \
  --spike True \
  --bfield True \
  --gamma True \
  --anisotropy True \
  --train_type pose \
  --use_fetal_inpainting False \
  --num_workers 8
