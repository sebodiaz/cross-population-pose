"""

This file generates synthetic training data for a pose estimation model.

"""


import os
import random
import scipy
import nibabel as nib
import numpy as np
from tqdm import tqdm
from copy import copy

def getMask(mask: np.array, kind='body'):
    assert kind in ['body', 'uterus'], 'kind must be either body or uterus'
    
    if kind == 'body':
        body = np.zeros(mask.shape)
        body[mask == 3] = 1
        body[mask == 4] = 1
        body[mask == 5] = 1
        return body.astype(bool)
    else:
        uterus = np.zeros(mask.shape)#; print(np.unique(mask))
        uterus[mask == 2] = 1
        uterus[mask == 3] = 1
        uterus[mask == 4] = 1
        uterus[mask == 5] = 1
        return uterus.astype(bool)
    
def getSynthI(volume: np.array, mask: np.array):
    body = np.zeros(volume.shape)
    body[mask == 3] = 1
    body[mask == 4] = 1
    body[mask == 5] = 1
    body = body.astype(bool)
    synth_I             = copy(volume)
    synth_I[body == 1]  = np.median(volume[mask == 2])
    synth_I[body == 1] += np.random.normal(0, 70, np.sum(body))
    return synth_I

def smoothUterus(synth_I: np.array, mask: np.array, uterus: np.array):
    synth_I[uterus == 1] = scipy.ndimage.gaussian_filter(synth_I, 1.5)[uterus == 1]
    return synth_I

# set the expectations for the data generation
num_train_data  = 1
files           = os.listdir('./clean-raw/')
final_coords    = [] #np.zeros((num_train_data, 3, 15))
key             = 15
# loop to generate synthetic data
successes = 0
for i in range(num_train_data):
    # inject some randomness
    ur    = random.choice(files)    # get a random file for the uterus
    bd    = random.choice(files)    # get a random file for the body

    # get data
    kp    = scipy.io.loadmat('/unborn/shared/SeboPoseLabel/' + bd.split('_')[0] + '.mat')['joint_coord'][int(bd.split('_')[1].split('.')[0])]  # Shape (3, 15)
    utv   = nib.load('./raw-bold-epi/' + ur).get_fdata()   # load uterus volume
    utm   = getMask(nib.load('./raw-bold-epi-outputs/' + ur).get_fdata(), 'uterus') # load uterus mask
    mmm   = nib.load('./raw-bold-epi-outputs/' + ur).get_fdata()
    
    bdv   = nib.load('./raw-bold-epi/' + bd).get_fdata()   # load body volume
    bdm   = getMask(nib.load('./raw-bold-epi-outputs/' + bd).get_fdata(), 'body'); print(np.sum(bdm)) # load body mask
    kpv   = np.zeros(bdv.shape)                         # create a blank volume for the keypoints
    for j in range(kp.shape[1]):
        y, x, z = kp[:, j]  # Extract coordinates
        x, y, z = int(round(x)), int(round(y)), int(round(z))  # Ensure integer indices
        
        # Define the 3x3x3 neighborhood
        for dx in range(-1, 2):  # [-1, 0, 1]
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    xn, yn, zn = x + dx, y + dy, z + dz
                    
                    # Ensure indices are within bounds
                    if 0 <= xn < kpv.shape[0] and 0 <= yn < kpv.shape[1] and 0 <= zn < kpv.shape[2]:
                        kpv[xn, yn, zn] = int(j + 1)  # Assign keypoint value
    
    # extract body
    bdv   = bdv * bdm # remove everything but the body from the body volume
    # save the body volume
    nib.save(nib.Nifti1Image(bdv, np.eye(4)), f'./train-data/ZM{key}/bd-{i}.nii.gz')
    
    # pad to the biggest size in each dimension
    biggest_x = max(bdv.shape[0], utv.shape[0])
    biggest_y = max(bdv.shape[1], utv.shape[1])
    biggest_z = max(bdv.shape[2], utv.shape[2])
    
    # pad the volumes
    bdv   = np.pad(bdv, ((0, biggest_x - bdv.shape[0]), (0, biggest_y - bdv.shape[1]), (0, biggest_z - bdv.shape[2])), mode='constant', constant_values=0)
    bdm   = np.pad(bdm, ((0, biggest_x - bdm.shape[0]), (0, biggest_y - bdm.shape[1]), (0, biggest_z - bdm.shape[2])), mode='constant', constant_values=0)
    utv   = np.pad(utv, ((0, biggest_x - utv.shape[0]), (0, biggest_y - utv.shape[1]), (0, biggest_z - utv.shape[2])), mode='constant', constant_values=0)
    utm   = np.pad(utm, ((0, biggest_x - utm.shape[0]), (0, biggest_y - utm.shape[1]), (0, biggest_z - utm.shape[2])), mode='constant', constant_values=0)
    kpv   = np.pad(kpv, ((0, biggest_x - kpv.shape[0]), (0, biggest_y - kpv.shape[1]), (0, biggest_z - kpv.shape[2])), mode='constant', constant_values=0)
    mmm   = np.pad(mmm, ((0, biggest_x - mmm.shape[0]), (0, biggest_y - mmm.shape[1]), (0, biggest_z - mmm.shape[2])), mode='constant', constant_values=0)
    
    # number of SE(3) transformations to attempt to fit the body to the uterus
    iterations      = 30
    upper_zoom      = 1.75  # was 1.5
    lower_zoom      = 0.75 # was 0.65
    previous_zoom   = 1
    for j in tqdm(range(iterations), ncols=25):
        # choose random zoom factor
        bodyzoom    = np.random.uniform(lower_zoom, 1)
        uteruszoom  = np.random.uniform(1, upper_zoom)
        
        # zoom the body and keypoints
        bdvz = scipy.ndimage.zoom(copy(bdv), bodyzoom, order=0) # TODO: zoom in different directions
        kpvz = scipy.ndimage.zoom(copy(kpv), bodyzoom, order=0)
        bdmz = scipy.ndimage.zoom(copy(bdm), bodyzoom, order=0)

        # pad to the shape of the uterus
        bdvz = np.pad(bdvz, ((0, utv.shape[0] - bdvz.shape[0]), 
                            (0, utv.shape[1] - bdvz.shape[1]), 
                            (0, utv.shape[2] - bdvz.shape[2])), mode='constant', constant_values=0)
        kpvz = np.pad(kpvz, ((0, utv.shape[0] - kpvz.shape[0]), 
                            (0, utv.shape[1] - kpvz.shape[1]), 
                            (0, utv.shape[2] - kpvz.shape[2])), mode='constant', constant_values=0)
        bdmz = np.pad(bdmz, ((0, utv.shape[0] - bdmz.shape[0]),
                            (0, utv.shape[1] - bdmz.shape[1]), 
                            (0, utv.shape[2] - bdmz.shape[2])), mode='constant', constant_values=0)
        #print(f'after pad bdvz: {bdvz.shape} | kpvz: {kpvz.shape}')
        
        # zoom uterus
        utmz = scipy.ndimage.zoom(copy(utm), uteruszoom, order=0)
        utvz = scipy.ndimage.zoom(copy(utv), uteruszoom, order=0)
        mmmz = scipy.ndimage.zoom(copy(mmm), uteruszoom, order=0)
        
        # crop uterus symmetrically
        cropx = (utmz.shape[0] - bdvz.shape[0]) // 2
        cropy = (utmz.shape[1] - bdvz.shape[1]) // 2
        cropz = (utmz.shape[2] - bdvz.shape[2]) // 2
        
        utmz = utmz[cropx:cropx + bdvz.shape[0], cropy:cropy + bdvz.shape[1], cropz:cropz + bdvz.shape[2]]
        utvz = utvz[cropx:cropx + bdvz.shape[0], cropy:cropy + bdvz.shape[1], cropz:cropz + bdvz.shape[2]]
        mmmz = mmmz[cropx:cropx + bdvz.shape[0], cropy:cropy + bdvz.shape[1], cropz:cropz + bdvz.shape[2]]
        
    
        # Transformation 0: Alignment of the center of masses to prepare for SE(3) transformations
        utv_center  = scipy.ndimage.center_of_mass(utmz)
        bmz_center  = scipy.ndimage.center_of_mass(bdmz)
        translation = np.array(utv_center) - np.array(bmz_center)
        bdvz        = scipy.ndimage.shift(bdvz, translation, order=0)
        bdmz        = scipy.ndimage.shift(bdmz, translation, order=0)
        kpvz        = scipy.ndimage.shift(kpvz, translation, order=0)
        
        # check if the body is inside the uterus
        grt_sum = np.sum(bdmz)
        mult    = utmz * bdmz#; print(np.sum(mult), grt_sum)
        pck     = 0
        
        # number of unique kpvz
        #print(f'unqiue: {np.unique(kpvz)}')
        
        if np.sum(mult) / grt_sum > 0.99:
            #print(f'found a match: {i}')
            
            # test 3 random rotations
            for k in range(3):
                # randomly rotate the body
                rotation = np.random.uniform(0, 2 * np.pi, 3)
                bdvz_rot = scipy.ndimage.rotate(bdvz, rotation[0], axes=(1, 2), reshape=False, order=0)
                bdmz_rot = scipy.ndimage.rotate(bdmz, rotation[0], axes=(1, 2), reshape=False, order=0)
                kpvz_rot = scipy.ndimage.rotate(kpvz, rotation[0], axes=(1, 2), reshape=False, order=0)
                
                # check if the body is inside the uterus
                grt_sum = np.sum(bdmz_rot)
                mult    = utmz * bdmz_rot#; print(np.sum(mult), grt_sum)
                
                if np.sum(mult) / grt_sum > 0.99:
                    #print(f'found a match after rotation: {i}')
                    
                    # generate the synthetic AF
                    synth = getSynthI(volume=utvz, mask=mmmz)
                    synth = smoothUterus(synth, mmmz, utmz)
                    # save the synth 
                    nib.save(nib.Nifti1Image(synth, np.eye(4)), f'./train-data/ZM{key}/AF.nii.gz')
                    
                    
                    
                    # new mask
                    bdmz_rot_mask = np.zeros(bdmz_rot.shape)
                    rotrot = copy(bdvz_rot)#; print(f'unqiue in rotrot: {np.unique(rotrot)}')
                    bdmz_rot_mask[rotrot > 5] = 1
                    bdmz_rot_mask = bdmz_rot_mask.astype(bool)
                    
                    synth[bdmz_rot_mask == 1] = bdvz_rot[bdmz_rot_mask == 1]
                    
                    # smooth the borders
                    borders = np.zeros(bdmz_rot_mask.shape, dtype=bool)
                    structure = np.ones((3,3,3))
                    dilated_mask = scipy.ndimage.binary_dilation(bdmz_rot_mask, structure, iterations=2)
                    borders = dilated_mask & ~bdmz_rot_mask
                    synth[borders == 1] = scipy.ndimage.gaussian_filter(synth, 3)[borders == 1]
                    
                    
                    
                    # take mean of the kpvz_rot
                    final_kpvz = np.zeros((3,15))
                    for kk in range(1, 16):
                        xs, ys, zs = np.where(kpvz_rot == kk)
                        xx = np.mean(xs)
                        yy = np.mean(ys)
                        zz = np.mean(zs)
                        
                        if xx == np.nan or yy == np.nan or zz == np.nan:
                            print(f'NaN found: {i}')
                            print(xx, yy, zz)
                            #continue
                        
                        final_kpvz[0, kk-1] = int(yy)
                        final_kpvz[1, kk-1] = int(xx)
                        final_kpvz[2, kk-1] = int(zz)
                    final_kpvz = final_kpvz[np.newaxis, ...]
                    final_coords.append(final_kpvz)

                    
                    nib.save(nib.Nifti1Image(synth, np.eye(4)), f'./train-data/ZM{key}/ZM{key}_{str(successes).zfill(4)}_{bd}.nii.gz')
                    final_coordss = np.concatenate(final_coords, axis=0)
                    #print(f'final dimensions: {final_coords.shape}')
                    scipy.io.savemat(f'./train-data/labels/ZM{key}.mat', {'joint_coord': final_coordss})
                    #nib.save(nib.Nifti1Image(kpvz_rot.astype(np.float32), np.eye(4)), f'./train-data/kp-{i}.nii.gz')
                    #scipy.io.savemat(f'./train-data/kp-{i}.mat', {'joint_coord': final_kpvz})
                    
                    
                    #nib.save(nib.Nifti1Image(bdvz_rot, np.eye(4)), f'./train-data/bd-{i}.nii.gz')
                    
                    # save rotated mask
                    #nib.save(nib.Nifti1Image(bdmz_rot_mask.astype(np.uint8), np.eye(4)), f'./train-data/bdm-{i}.nii.gz')
                    
                    #nib.save(nib.Nifti1Image(utmz.astype(np.uint8), np.eye(4)), f'./train-data/ut-{i}.nii.gz')
                    #nib.save(nib.Nifti1Image(kpvz_rot, np.eye(4)), f'./train-data/kp-{i}.nii.gz')
                    #nib.save(nib.Nifti1Image(mmmz.astype(np.uint8), np.eye(4)), f'./train-data/mm-{i}.nii.gz')
                    pck = 1
                    successes += 1
                    break
                elif k == 2:
                    #print(f'no match found after 3 rotations: {i}')
                    
                    # generate the synthetic AF
                    synth = getSynthI(volume=utvz, mask=mmmz)
                    synth = smoothUterus(synth, mmmz, utmz)
                    
                    # new mask
                    bdmz_rot_mask = np.zeros(bdmz.shape)
                    rotrot = copy(bdvz)#; print(f'unqiue in rotrot: {np.unique(rotrot)}')
                    bdmz_rot_mask[rotrot > 5] = 1
                    bdmz_rot_mask = bdmz_rot_mask.astype(bool)
                    
                    synth[bdmz_rot_mask == 1] = bdvz[bdmz_rot_mask == 1]
                    
                    # smooth the borders
                    borders = np.zeros(bdmz_rot_mask.shape, dtype=bool)
                    structure = np.ones((3,3,3))
                    dilated_mask = scipy.ndimage.binary_dilation(bdmz_rot_mask, structure, iterations=2)
                    borders = dilated_mask & ~bdmz_rot_mask
                    synth[borders == 1] = scipy.ndimage.gaussian_filter(synth, 3)[borders == 1]
                    
                    
                    # take mean of the kpvz_rot
                    final_kpvz = np.zeros((3,15))
                    for kk in range(1, 16):
                        xs, ys, zs = np.where(kpvz_rot == kk)
                        xx = np.mean(xs)
                        yy = np.mean(ys)
                        zz = np.mean(zs)
                        
                        if xx == np.nan or yy == np.nan or zz == np.nan:
                            print(f'NaN found: {i}')
                            print(xx, yy, zz)
                        
                        
                        
                        final_kpvz[0, kk-1] = int(yy)
                        final_kpvz[1, kk-1] = int(xx)
                        final_kpvz[2, kk-1] = int(zz)
                    final_kpvz = final_kpvz[np.newaxis, ...]
                    final_coords.append(final_kpvz)
                    nib.save(nib.Nifti1Image(synth, np.eye(4)), f'./train-data/ZM{key}/ZM{key}_{str(successes).zfill(4)}_{bd}.nii.gz')
                    successes += 1
                    pck = 1
                    
                    
                    
                    final_coordss = np.concatenate(final_coords, axis=0)
                    #print(f'final dimensions: {final_coords.shape}')
                    scipy.io.savemat(f'./train-data/labels/ZM{key}.mat', {'joint_coord': final_coordss})
                                    
                                        
                    
                    
                    break
            
        if pck == 0:
            pass
            #print(f'no match found: {i}')
        elif pck == 1:
            #print('terminating early')
            break
        
# save the final coordinates
#final_coords = np.concatenate(final_coords, axis=0)
#print(f'final dimensions: {final_coords.shape}')
#scipy.io.savemat(f'./train-data/labels/ZM1.mat', {'joint_coord': final_coords})
                