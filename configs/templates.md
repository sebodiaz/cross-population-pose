# Examples of scripts for testing, training, inference, and continuation

>[!warning]
>Ensure you read over the `options.py` file to entertain all possible options.

## Testing
```bash
python main.py --run_name=baseline-E40\
               --gpu_ids=1\
               --stage=test\
               --use_continue=True\
               --continue_path='/unborn/sdd/PyTorchPose/runs/baseline/checkpoints/checkpoint_40.pth'\
               --error_threshold=3\
```

## Training
```bash
python main.py --run_name=augmentation\
               --gpu_ids=1\
               --stage=train\
               --rot=True\
               --lr_scheduler=linear\
               --zoom_prob=0.5\
               --zoom_factor=0.65\
               --custom_augmentation=True\
               --augmentation_prob=0.9\
               --seed=42\ # if you want to pause and resume training
```

## Inference

There are two types of inference scripts:
+ individual: applicable to a single time-series acquisition
+ loop: applicable to multiple time-series acquisitions


### Individual inference

```bash
python main.py --name=infer\
               --gpu_ids=0\
               --stage=inference\
               --rawdata_path='/unborn/shared/fetal_pose/fetalEPI/021218L'\
               --use_continue=True\
               --continue_path='/unborn/sdd/PyTorchPose/runs/baseline/checkpoints/checkpoint_40.pth'\
               --output_path='./inference_output'\
               --output_vis=True\
               --label_name='021218L'\
               --index_type=1\
```

### Multiple inference

```bash
# define the directory
patient_directory="/unborn/shared/fetal_pose/fetalEPI"
# define variables
gpu_id=0
continue_path="/unborn/sdd/PyTorchPose/runs/baseline/checkpoints/checkpoint_40.pth"
output_path="./inference_output/"
stage="inference"
index_type=1
# loop through the patient directories
# only print the patient name for now
for patient in $(ls $patient_directory); do

    # run the inference
    python main.py --name=infer\
                   --gpu_ids=$gpu_id\
                   --stage=$stage\
                   --rawdata_path=$patient_directory/$patient\
                   --use_continue=True\
                   --continue_path=$continue_path\
                   --output_path=$output_path\
                   --output_vis=True\
                   --label_name=$patient\
                   --index_type=$index_type
done
```

## Continuation
Essentially, all you need to do is copy and paste the original `train.sh` script here and add three things:
1. set `use_continue` to **True**
2. set `continue_path` to the correct path
3. set `run_id` to the listed WandB run id
_Optionally_: ensure the seed is identical

```bash
python main.py --run_name=augmentation\
               --gpu_ids=1\
               --stage=train\
               --rot=True\
               --lr_scheduler=linear\
               --zoom_prob=0.5\
               --zoom_factor=0.65\
               --custom_augmentation=True\
               --augmentation_prob=0.9\
               --seed=42\
               --use_continue=True\
               --continue_path='..path/checkpoints/E##.pth'\
               --run_id=run_id\
```

