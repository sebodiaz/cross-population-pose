"""

This file fetches feasible masks for training by checking if all keypoints are within the body mask.

"""



import os
import nibabel as nib
import scipy
import random
import numpy as np

def getMask(mask: np.array, kind='body'):
    assert kind in ['body', 'uterus'], 'kind must be either body or uterus'
    
    if kind == 'body':
        body = np.zeros(mask.shape)
        body[mask == 3] = 1
        body[mask == 4] = 1
        body[mask == 5] = 1
        body = body.astype(bool)
        return body
    else:
        uterus = np.zeros(mask.shape)
        uterus[mask == 2] = 1
        uterus[mask == 3] = 1
        uterus[mask == 4] = 1
        uterus[mask == 5] = 1
        uterus = uterus.astype(bool)
        return uterus

# these are the subject folders (or, equivalently, subject IDs) that we will use for training
data_train = [
    '072017L', '072017S', '110217L', '031615', '031616', '043015', 
    '052218L', '052218S', '013118L', '013118S', '121517a', '121517b',
    '032318a', '032318b', '111017L', '111017S', '010918L', '010918S',
    '021218L', '021218S', '031317L', '031317T', '062817L', '062817S',
    '103017a', '103017b', '013018L', '013018S', '051718L', '051718S',
    '053017', '082917a', '082917b', '052418L', '052418S', '071717L',
    '071717S', '091917L', '091917S', '022318L', '022318S', '053117L',
    '053117S', '083017L', '083017S', '032318c', '032318d',
    '082117L', '082117S', '032217'
]


import os
import random
import nibabel as nib
import scipy.io
import numpy as np

folder_path = "./raw-bold-epi-outputs/"
files = sorted(os.listdir(folder_path))  # Sort to maintain order

for _ in range(len(files)):  # Loop for the number of files
    file = random.choice(files)  # Sample a random file
    patient_id = file.split('_')[0]  # Extract patient ID

    if patient_id in data_train:  # Check if patient ID is in data_train
        # Load the patient masks
        volume = nib.load(os.path.join('./raw-bold-epi/', file)).get_fdata()
        mask = nib.load(os.path.join('./raw-bold-epi-outputs/', file)).get_fdata()

        # Extract the index from the filename
        idx = int(file.split('_')[1].split('.')[0])

        # Load the patient keypoints
        keypoints = scipy.io.loadmat(f'/unborn/shared/SeboPoseLabel/{patient_id}.mat')['joint_coord'][idx]  # Shape (3, 15)
        
        # Convert the mask to binary (body region)
        body = getMask(mask, kind='body').astype(np.uint8)

        # Check if all keypoints are inside the body mask
        all_inside = True  # Flag to track if all keypoints are inside
        for j in range(keypoints.shape[1]):  # Loop over 15 keypoints
            y, x, z = keypoints[:, j]
            x, y, z = int(round(x)), int(round(y)), int(round(z))

            if (0 <= x < body.shape[0]) and (0 <= y < body.shape[1]) and (0 <= z < body.shape[2]):
                if body[x, y, z] == 0:
                    all_inside = False
                    #print(f'Patient {patient_id}, Keypoint {j} at {keypoints[:, j]} is NOT within the mask.')
            else:
                all_inside = False
                #print(f'Patient {patient_id}, Keypoint {j} at {keypoints[:, j]} is OUT OF BOUNDS.')

        if all_inside:
            # save the raw volume
            idx = str(idx).zfill(4)
            nib.save(nib.Nifti1Image(volume, np.eye(4)), f'./clean-raw/{patient_id}_{idx}.nii.gz')
            nib.save(nib.Nifti1Image(mask, np.eye(4)), f'./clean-masks/{patient_id}_{idx}.nii.gz')
            
            
            print(f'All keypoints inside body mask for {patient_id}_{idx}')
