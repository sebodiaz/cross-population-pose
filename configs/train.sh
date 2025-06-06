python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py  --run_name=tmp\
                --num_nodes=1\
                --num_gpus=1\
                --use_amp=True\
                --unet_type=small\
                --optimizer=adam\
                --depth=5\
                --nFeat=16\
                --stage=train\
                --loss=mse\
                --mag=10\
                --batch_size=16\
                --crop_size=64\
                --lr=2e-4\
                --nJoints=17\
                --epochs=1000\
                --lr_scheduler=linear\
                --dataset_size=8000\
                --augmentation_prob=0.9\
                --save_path=/data/vision/polina/users/sebodiaz/projects/pose/runs/\
                --num_workers=8\
                --seed=45\
                --custom_augmentation=True\
                --zoom=0.75\
                --bfield=True\
                --gamma=True\
                --anisotropy=True\
                --spike=True\
                --noise=True\
                --label_path=/data/vision/polina/projects/fetal/common-data/pose/SeboPoseLabel\
                --rawdata_path=/data/vision/polina/projects/fetal/common-data/pose/epis\
                --csail=True\