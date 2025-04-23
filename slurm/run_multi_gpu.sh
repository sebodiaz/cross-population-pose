#!/bin/bash
## SLURM Variables:
#SBATCH --job-name tm_tmp_tmultigpu
#SBATCH --output=/data/vision/polina/users/sebodiaz/projects/pose_fin/logs/pose.out
#SBATCH -e /data/vision/polina/users/sebodiaz/projects/pose/errors/%x-%j.err
#SBATCH -o /data/vision/polina/users/sebodiaz/projects/pose/outputs/%x-%j.out
#SBATCH --partition=polina-a6000
#SBATCH --ntasks-per-node=2                 # needs to match opts.num_gpus
#SBATCH --gres=gpu:2                        # needs to mirror `ntasks-per-node`
#SBATCH --nodes=1                           # needs to match opts.num_nodes
#SBATCH --qos=vision-polina-main
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=6-00:00:00

# activate virtual environment
source /data/vision/polina/users/sebodiaz/miniconda3/bin/activate pose
export PYTHONPATH="/data/vision/polina/users/sebodiaz/projects/pose_fin:${PYTHONPATH}"
#bash /data/vision/polina/users/sebodiaz/projects/pose_fin/configs/train-temporal.sh

srun python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py  --run_name=tmp\
                --num_nodes=1\
                --num_gpus=2\
                --use_amp=True\
                --unet_type=small\
                --optimizer=adam\
                --stage=train\
                --loss=mse\
                --mag=10\
                --batch_size=16\
                --crop_size=64\
                --lr=2e-4\
                --epochs=1000\
                --lr_scheduler=linear\
                --train_type=offline\
                --dataset_size=8000\
                --augmentation_prob=0.9\
                --save_path=/data/vision/polina/users/sebodiaz/projects/pose/runs/\
                --num_workers=8\
                --seed=45\
                --custom_augmentation=True\
                --ablation=True\
                --zoom=0.75\
                --bfield=True\
                --gamma=True\
                --anisotropy=True\
                --spike=True\
                --noise=True\
                --label_path=/data/vision/polina/projects/fetal/common-data/pose/SeboPoseLabel\
                --rawdata_path=/data/vision/polina/projects/fetal/common-data/pose/epis\
                --csail=True\