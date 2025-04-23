python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py  --run_name=tm_tamp_tamix_tmp_tzw_32ngf_ncat\
                --nFeat=32\
                --anatomix=True\
                --num_nodes=1\
                --num_gpus=1\
                --use_amp=True\
                --use_zoom_womb=True\
                --model_name=small_unet\
                --optimizer=adam\
                --stage=train\
                --loss=mse\
                --mag=10\
                --batch_size=8\
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
                --csail=True\
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
                
                #--continue_path=/data/vision/polina/users/sebodiaz/projects/pose/runs/tm_tt2_fzw_ttsm/checkpoints/latest.pth\
                #--run_id=17p8hcgs\
                #--temporal=3\
                #--four_dim=True\
                #--tsm=True\




                