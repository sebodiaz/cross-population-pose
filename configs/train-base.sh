python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py --run_name=baseline-rebuttal16-small\
               --gpu_ids=1\
               --optimizer=adamw\
               --stage=train\
               --rot=True\
               --lr=1e-3\
               --batch_size=16\
               --crop_size=64\
               --epochs=1000\
               --baseline=True\
               --lr_scheduler=cosine\
               --junshen_scale=0.2\
               --zoom=0.5\
               --depth=4\
               --nFeat=16\
               --model_name=small_unet\
               --zoom_factor=1.5\
               --rawdata_path=/data/vision/polina/projects/fetal/common-data/pose/epis\
               --label_path=/data/vision/polina/projects/fetal/common-data/pose/SeboPoseLabel\
               --dataset_size=8000\
               --csail=True\
               --num_workers=8\
               --save_path=/data/vision/polina/users/sebodiaz/data/runs/\
               --seed=54\