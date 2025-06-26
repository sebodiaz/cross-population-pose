"""

This file synthesizes training data for a pose estimation model by generating synthetic images and masks.
It was not used in the original training pipeline but is useful for prototyping and testing purposes.


"""
import os
import numpy as np
from copy import copy
import scipy
import nibabel as nib
import scipy.ndimage

def getPatient(path: 'str'):
    files = os.listdir(path)
    file  = np.random.choice(files)
    return file

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

def getSynthI(volume: np.array, mask: np.array, body: np.array):
    synth_I             = copy(volume)
    synth_I[body == 1]  = np.median(volume[mask == 2])
    synth_I[body == 1] += np.random.normal(0, 70, np.sum(body))
    return synth_I

def smoothUterus(synth_I: np.array, mask: np.array, uterus: np.array):
    synth_I[uterus == 1] = scipy.ndimage.gaussian_filter(synth_I, 1.5)[uterus == 1]
    return synth_I

def ZoomVolume(volume: np.array, mask: np.array, zf=0.75):
    zoomed_volume   = scipy.ndimage.zoom(volume, zf, order=0)
    zoomed_mask     = scipy.ndimage.zoom(mask, zf, order=0)
    zoomed_volume   = zoomed_volume * zoomed_mask
    return zoomed_volume, zoomed_mask

def padVolume(volume: np.array, mask: np.array):
    if volume.shape != mask.shape:
        pad = (volume.shape[0] - mask.shape[0]) // 2
        zpad = (volume.shape[2] - mask.shape[2]) // 2
        mask = np.pad(mask, ((pad, pad), (pad, pad), (zpad, zpad)), 'constant', constant_values=0)
    return mask

def completePadding(volume: np.array, volume1: np.array):
    pad     = (volume.shape[0] - volume1.shape[0]) // 2
    zpad    = (volume.shape[2] - volume1.shape[2]) // 2
    volume1 = np.pad(volume1, ((pad, pad), (pad, pad), (zpad, zpad)), 'constant', constant_values=0)
    volume1 = padVolume(volume, volume1)
    
    return volume1

def apply_random_rotation(zoomed_volume, zoomed_mask, uterus_mask):
    try:
        # Generate a random angle between -15 and 15 degrees
        theta = np.random.uniform(-30,30)
        
        
        # shift the zoomed_mask so that the center of mass is at the center of mass of the uterus mask
        
        # translate it
        #zoomed_volume_rot = scipy.ndimage.shift(zoomed_volume, translation, order=0)
        #zoomed_mask_rot = scipy.ndimage.shift(zoomed_mask, translation, order=0)
        translation = np.random.uniform(-5, 5, 3)
    
        zoomed_volume_rot = scipy.ndimage.shift(zoomed_volume, translation, order=0)
        zoomed_mask_rot = scipy.ndimage.shift(zoomed_mask, translation, order=0)
        
        # Apply the rotation to both zoomed_volume and zoomed_mask
        zoomed_volume_rot = scipy.ndimage.rotate(zoomed_volume_rot, theta, axes=(0), reshape=False, order=0)
        zoomed_mask_rot = scipy.ndimage.rotate(zoomed_mask_rot, theta, axes=(0), reshape=False, order=0)
        
        
        
        

        # Ensure that the rotated mask does not contain any values outside the uterus_mask
        #print(f'shapes of the rotated mask and uterus mask: {zoomed_mask_rot.shape}, {uterus_mask.shape}')
        if np.any(zoomed_mask_rot[uterus_mask == 0] == 1):
            
            raise ValueError("Rotated mask contains values outside the uterus mask")
        print(f"Rotation angle: {theta}° | Translation: {translation}")
        
        #print("Rotation successful!")
        return True, zoomed_mask_rot, zoomed_volume_rot  # Return success and the angle used
    except Exception as e:
        #print(f"Rotation failed with error: {e}")
        return False, None, None  # Return failure and no angle


import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import fftconvolve

def sample_se3_transformation(A, B, max_attempts=100):
    """
    Sample a valid SE(3) transformation (rotation + translation) for B such that B remains inside A.
    """
    # Ensure A and B are binary arrays
    A = A.astype(bool)
    B = B.astype(bool)
    
    # Get coordinates of B
    b_coords = np.argwhere(B)
    center = np.array(B.shape) // 2  # Center of B
    
    for _ in range(max_attempts):
        # Sample a random rotation
        R = Rotation.random()
        rotation_matrix = R.as_matrix()
        
        # Rotate B's coordinates
        centered_coords = b_coords - center
        rotated_coords = R.apply(centered_coords)
        rotated_coords = np.round(rotated_coords + center).astype(int)
        
        # Filter out-of-bounds coordinates
        valid = np.all((rotated_coords >= 0) & (rotated_coords < np.array(B.shape)), axis=1)
        if not valid.any():
            continue  # Skip if rotation moves B entirely out of bounds
        
        rotated_coords = rotated_coords[valid]
        rotated_B = np.zeros_like(B)
        rotated_B[tuple(rotated_coords.T)] = 1
        
        # Compute valid translations using convolution
        n = np.sum(rotated_B)
        if n == 0:
            continue
        
        kernel = rotated_B[::-1, ::-1, ::-1]  # Flip for convolution
        conv = fftconvolve(A.astype(int), kernel.astype(int), mode='valid')
        valid_ts = np.argwhere(conv == n)
        
        if len(valid_ts) > 0:
            t = valid_ts[np.random.choice(len(valid_ts))]
            return rotation_matrix, t
        
    raise RuntimeError("Failed to find a valid transformation after max attempts.")


from scipy.interpolate import RegularGridInterpolator

def rotate_volume(volume, rotation_matrix):
    """
    Rotates a 3D volume using a 3x3 rotation matrix.
    
    Args:
        volume (np.ndarray): The input 3D volume (e.g., a binary mask or grayscale image).
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.
    
    Returns:
        np.ndarray: The rotated volume.
    """
    # Get the shape of the input volume
    depth, height, width = volume.shape
    
    # Create a grid of coordinates for the original volume
    z, y, x = np.meshgrid(
        np.arange(depth),
        np.arange(height),
        np.arange(width),
        indexing='ij'
    )
    
    # Stack coordinates into a (N, 3) array
    coords = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)
    
    # Center the coordinates for rotation
    center = np.array([depth // 2, height // 2, width // 2])
    centered_coords = coords - center
    
    # Apply the rotation matrix
    rotated_coords = np.dot(centered_coords, rotation_matrix.T)
    
    # Shift back to the original coordinate system
    rotated_coords += center
    
    # Interpolate the rotated volume
    interpolator = RegularGridInterpolator(
        (np.arange(depth), np.arange(height), np.arange(width)),
        volume,
        method='linear',  # Use 'nearest' for binary masks
        bounds_error=False,
        fill_value=0  # Fill value for out-of-bounds coordinates
    )
    
    # Reshape the rotated coordinates for interpolation
    rotated_volume = interpolator(rotated_coords)
    rotated_volume = rotated_volume.reshape((depth, height, width))
    
    return rotated_volume






if __name__ == "__main__":
    # test
    path    = 'raw-bold-epi-outputs/'
    file    = getPatient(path); print(f'File: {file}')
    
    # get the patient name
    acquisition = file.split('.')[0]
    patient_name, idx = acquisition.split('_')[0], acquisition.split('_')[1]
    idx = int(idx)
    
    # fetch the keypoint data from the server
    coords = scipy.io.loadmat(f'/unborn/shared/SeboPoseLabel/{patient_name}.mat')['joint_coord'][idx]
    
    print(f'Patient name: {patient_name}, idx: {idx}')
    mask    = nib.load(f'raw-bold-epi-outputs/{file}').get_fdata()
    volume  = nib.load(f'raw-bold-epi/{file}').get_fdata()
    
    
    # get the body and uterus masks
    body    = getMask(mask, kind='body')
    body    = scipy.ndimage.binary_dilation(body, iterations=2)
    uterus  = getMask(mask, kind='uterus')

    # get the synthetic image
    synth_I = getSynthI(volume, mask, body)
    
    # smooth the uterus
    synth_I = smoothUterus(synth_I, mask, uterus)
    synth_I = scipy.ndimage.zoom(synth_I, 1.75, order=0)
    uterus  = scipy.ndimage.zoom(uterus, 1.75, order=0)
    
    # crop the volume to match the original volume shape
    cs_x    = (synth_I.shape[0] - volume.shape[0]) // 2
    cs_y    = (synth_I.shape[1] - volume.shape[1]) // 2
    cs_z    = (synth_I.shape[2] - volume.shape[2]) // 2
    synth_I = synth_I[cs_x:cs_x+volume.shape[0], cs_y:cs_y+volume.shape[1], cs_z:cs_z+volume.shape[2]]
    uterus  = uterus[cs_x:cs_x+volume.shape[0], cs_y:cs_y+volume.shape[1], cs_z:cs_z+volume.shape[2]]
    

    # zoom the volume
    zf                          = 0.85
    zoomed_volume, zoomed_mask  = ZoomVolume(volume, body, zf=zf)
    coords                      = coords * zf
    coords_volume = np.zeros(zoomed_volume.shape)
    for i in range(coords.shape[1]):
        coords_volume[int(coords[1, i]), int(coords[0, i]), int(coords[2, i])] = int(10) #int(i + 1)
    
    # pad the zoomed volume
    zoomed_volume = completePadding(volume, zoomed_volume)
    zoomed_mask   = completePadding(volume, zoomed_mask)
    zoomed_coords = completePadding(volume, coords_volume)
    
    if zoomed_volume.shape != volume.shape:
        distancez       = volume.shape[2] - zoomed_mask.shape[2]
        zoomed_mask     = np.pad(zoomed_mask, ((0, 0), (0, 0), (0, distancez)), 'constant', constant_values=0)
        zoomed_volume   = np.pad(zoomed_volume, ((0, 0), (0, 0), (0, distancez)), 'constant', constant_values=0)
        zoomed_coords   = np.pad(zoomed_coords, ((0, 0), (0, 0), (0, distancez)), 'constant', constant_values=0)
    print(f'Shapes of the original volume and mask: {volume.shape}, {body.shape}, {zoomed_volume.shape}, {zoomed_mask.shape}, {uterus.shape}')
    

    
    # apply the random rotations to the zoomed volume and mask
    zoomed_maskk    = zoomed_mask
    zoomed_volumee  = zoomed_volume
    
    # insert the zoomed volume into the synth_I
    synth_I[zoomed_maskk == 1] = zoomed_volumee[zoomed_maskk == 1]
    
    # smooth the borders
    borders               = np.zeros(mask.shape, dtype=bool)
    structure             = np.ones((3,3,3))
    dilated_mask          = scipy.ndimage.binary_dilation(zoomed_mask, structure, iterations=2)
    borders               = dilated_mask & ~zoomed_mask
    synth_I[borders == 1] = scipy.ndimage.gaussian_filter(synth_I, 3)[borders == 1]
    
    
    # get number of unique values in zoomed_coords
    unique_values = np.unique(zoomed_coords)
    print(f'Number of unique values in zoomed_coords: {len(unique_values) - 1}')
    
    # save the synthetic image
    nib.save(nib.Nifti1Image(synth_I, np.eye(4)), 'synth_II.nii.gz')
    
    # save the masks
    #nib.save(nib.Nifti1Image(body, np.eye(4)), 'body.nii.gz')
    #nib.save(nib.Nifti1Image(uterus.astype(np.float32), np.eye(4)), 'uterus.nii.gz')
    #nib.save(nib.Nifti1Image(zoomed_mask.astype(np.float32), np.eye(4)), 'zoomed_mask.nii.gz')
    nib.save(nib.Nifti1Image(zoomed_coords.astype(np.float32), np.eye(4)), 'label.nii.gz')
        