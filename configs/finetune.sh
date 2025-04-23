python /data/vision/polina/users/sebodiaz/projects/pose/main.py  --run_name=finetune_longer\
                --gpu_ids=0\
                --optimizer=adam\
                --stage=train\
                --loss=mse\
                --batch_size=16\
                --crop_size=64\
                --lr=3e-5\
                --sigma=1.5\
                --epochs=300\
                --lr_scheduler=linear\
                --train_type=offline\
                --unet_type=small\
                --label_path=/data/vision/polina/users/sebodiaz/data/clinical/labels\
                --rawdata_path=/data/vision/polina/users/sebodiaz/data/clinical/volumes\
                --save_path=/data/vision/polina/users/sebodiaz/projects/pose/runs/\
                --num_workers=8\
                --proc_type=sebo\
                --csail=True\
                --seed=45\
                --custom_augmentation=False\
                --continue_path=/data/vision/polina/users/sebodiaz/projects/pose/runs/agro_1k_small_zw/checkpoints/latest.pth\
                --stage=finetune\

                