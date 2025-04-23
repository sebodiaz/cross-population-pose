python /data/vision/polina/users/sebodiaz/projects/pose/main.py  --run_name=repulsive\
                --unet_type=small\
                --gpu_ids=0\
                --optimizer=adam\
                --stage=train\
                --loss=repulsive\
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
                --data_partition_file=/data/vision/polina/users/sebodiaz/projects/pose/data_partition.yml\
                --num_workers=8\
                --proc_type=sebo\
                --csail=True\
                --seed=45\
                --custom_augmentation=True\
                --use_zoom_womb=True\
                --ablation=True\
                --zoom=0.75\
                --bfield=True\
                --gamma=True\
                --anisotropy=True\
                --spike=True\
                --noise=True\
                --label_path=/data/vision/polina/projects/fetal/common-data/pose/SeboPoseLabel\
                --rawdata_path=/data/vision/polina/projects/fetal/common-data/pose/epis\




                