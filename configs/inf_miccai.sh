# Define the directory and other variables
patient_directory="/data/vision/polina/users/sebodiaz/data/old_clinical/volumes" # no "/" at the end is important
gpu_id=0
rname="tm_tamp_tmp_tzw_32ngf"
continue_path="/data/vision/polina/users/sebodiaz/projects/pose/runs/${rname}/checkpoints/latest.pth"
output_path="/data/vision/polina/users/sebodiaz/projects/pose_fin/results/inference"
stage="inference"
index_type=1
name="${rname}1"
# Populate the list of patient names from the directory
patients=('00' '01' '03' '04' '05' '06' '07'
'09' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' 
'20' '21' '22' '23' '24' '25' '26' '27' '29' '30' '31'
'32' '33' '34' '35' '36' '37' '38' '39')

# Loop through the predefined list of patient names
for patient in "${patients[@]}"; do

    # Construct label name with prefix
    label_name="${patient}"
    
    # Echo the patient name
    echo "Processing patient: $patient"

    # Run the inference for each patient
    python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py --run_name=$name\
                   --gpu_ids=0\
                   --stage=inference\
                   --rawdata_path=$patient_directory/$patient/\
                   --continue_path=$continue_path\
                   --output_path=$output_path\
                   --label_name=$label_name\
                   --index_type=$index_type\
                   --csail=True\
                   --model_name=small_unet\
                   --nFeat=32\

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
    python /data/vision/polina/users/sebodiaz/projects/pose_fin/main.py --run_name=$name\
                   --gpu_ids=0\
                   --stage=inference\
                   --rawdata_path=$patient_directory/$patient/\
                   --continue_path=$continue_path\
                   --output_path=$output_path\
                   --label_name=$label_name\
                   --index_type=$index_type\
                   --csail=True\
                   --model_name=small_unet\
                   --nFeat=32\


done