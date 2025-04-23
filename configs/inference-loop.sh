# Define the list of patient names
patients=('021218L' '022618' '041818' '052516' '053117S' '072017L' '082917a' 
          '083017L' '083115' '090517L' '091917L' '100317S' '110214' '120717')

# Define the directory and other variables
patient_directory="/unborn/shared/fetal_pose/fetalEPI"
gpu_id=0
continue_path="/unborn/sdd/PyTorchPose/runs/augmentation/checkpoints/E160.pth"
output_path="./results/inference/"
stage="inference"
index_type=1
prefix="aug"  # Define your desired prefix
top_k=4
# Loop through the predefined list of patient names
for patient in "${patients[@]}"; do

    # Construct label name with prefix
    label_name="${prefix}_${patient}"
    
    # Echo the patient name
    echo "Processing patient: $patient"

    # Run the inference for each patient
    python main.py --run_name=infer\
                   --gpu_ids=$gpu_id\
                   --stage=$stage\
                   --rawdata_path=$patient_directory/$patient\
                   --use_continue=True\
                   --continue_path=$continue_path\
                   --output_path=$output_path\
                   --label_name=$label_name\
                   --index_type=$index_type\
                   --top_k=$top_k\

done
