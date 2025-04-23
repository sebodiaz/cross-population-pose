# Define the directory and other variables
patient_directory="/data/vision/polina/projects/fetal/common-data/pose/epis" # no "/" at the end is important
gpu_id=0
network="tm_tmp_trepulse_mean"
continue_path="/data/vision/polina/users/sebodiaz/projects/pose/runs/${network}/checkpoints/latest.pth"
output_path="/data/vision/polina/users/sebodiaz/projects/pose_fin/results/inference"
stage="inference"
index_type=1
# Populate the list of patient names from the directory
patients=('100317L' '100317S' '120717' '090517L' '050318L' '051817'
    '062117' '102617' '092817L' '040218' '041017' '101317'
    '082517L' '041318L')
network="tm_tmp_trepulse_mean"
# Loop through the predefined list of patient names
for patient in "${patients[@]}"; do

    # Construct label name with prefix
    label_name="${patient}"
    
    # Echo the patient name
    echo "Processing patient: $patient"

    # Run the inference for each patient
    python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py --run_name=$network\
                   --gpu_ids=0\
                   --stage=inference\
                   --rawdata_path=$patient_directory/$patient/\
                   --continue_path=$continue_path\
                   --output_path=$output_path\
                   --label_name=$label_name\
                   --index_type=$index_type\
                   --model_name=small_unet\
                   --depth=4\
                   --csail=True\
                   --nFeat=16\
                   
done
