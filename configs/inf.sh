# Define the directory and other variables
patient_directory="/data/vision/polina/users/sebodiaz/data/clinical/volumes" # no "/" at the end is important
gpu_id=0
rname="finetune_e_300"
continue_path="/data/vision/polina/users/sebodiaz/projects/pose/runs/${rname}/checkpoints/latest.pth"
output_path="/data/vision/polina/users/sebodiaz/projects/pose/results/inference"
stage="inference"
index_type=1
name="${rname}1"
# Populate the list of patient names from the directory
patients=('04640' '10034' '14130' '06938' '10035' '04042' '14133' '04043' '10535' '16540' '12243' '08637' '14234' '10331' '14027' '11941' '10120' '13831' '06941' '09534' '05241' '06944' '05242' '15044' '15736' '10935' '10628' '16729' '08259' '09420' '01736' '05222' '10835' '08539' '12832' '01735' '01734')

# Loop through the predefined list of patient names
for patient in "${patients[@]}"; do

    # Construct label name with prefix
    label_name="${patient}"
    
    # Echo the patient name
    echo "Processing patient: $patient"

    # Run the inference for each patient
    python /data/vision/polina/users/sebodiaz/projects/pose/main.py --run_name=$name\
                   --gpu_ids=0\
                   --stage=inference\
                   --rawdata_path=$patient_directory/$patient/volumes/\
                   --continue_path=$continue_path\
                   --output_path=$output_path\
                   --label_name=$label_name\
                   --index_type=$index_type\
                   --unet_type=small\
                   --csail=True\

done


# Define the directory and other variables
patient_directory="/data/vision/polina/projects/fetal/common-data/pose/epis" # no "/" at the end is important

# Populate the list of patient names from the directory
patients=('100317L' '100317S' '120717' '090517L' '050318L' '051817'
    '062117' '102617' '092817L' '040218' '041017' '101317'
    '082517L' '041318L')

# Loop through the predefined list of patient names
for patient in "${patients[@]}"; do

    # Construct label name with prefix
    label_name="${patient}"
    
    # Echo the patient name
    echo "Processing patient: $patient"

    # Run the inference for each patient
    python /data/vision/polina/users/sebodiaz/projects/pose/main.py --run_name=$name\
                   --gpu_ids=0\
                   --stage=inference\
                   --rawdata_path=$patient_directory/$patient/\
                   --continue_path=$continue_path\
                   --output_path=$output_path\
                   --label_name=$label_name\
                   --index_type=$index_type\
                   --unet_type=small\
                   --csail=True\

done