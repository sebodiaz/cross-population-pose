"""
Dataset class for fetal pose estimation.

Originally written by Sebo 01/07/2025. A lot of code is copy/pasted from Molin Zhang and Junshen Xu.

"""

# Import necessary libraries
import scipy.io as sio
import os
import nibabel as nib
import numpy as np
from random import randint, choice, uniform, random
from math import ceil
from scipy.ndimage import zoom
import torch
import torchio as tio
import numpy as np
import monai.transforms as transforms
from monai.config import KeysCollection
from typing import Optional, Dict, Any
from einops import rearrange

# Training, validation, and testing subjects
data_train = [
    '072017L', '072017S', '110217L', '031615', '031616', '043015', 
    '052218L', '052218S', '013118L', '013118S', '121517a', '121517b',
    '032318a', '032318b', '111017L', '111017S', '010918L', '010918S',
    '021218L', '021218S', '031317L', '031317T', '062817L', '062817S',
    '103017a', '103017b', '013018L', '013018S', '051718L', '051718S',
    '053017', '082917a', '082917b', '052418L', '052418S', '071717L',
    '071717S', '091917L', '091917S', '022318L', '022318S', '053117L',
    '053117S', '083017L', '083017S', '032318c', '032318d',
    '082117L', '082117S', '032217',]

data_finetune = ['04640', '10034', '14130', '06938', '10035',
                 '04042', '14133', '04043', '10535', '16540',
                 '12243', '08637', '14234', '10331', '14027',
                 '11941', '10120', '13831', '06941', '09534',
                 '05241', '06944', '05242', '15044', '15736',
                 '10935', '10628', '16729', '08259', '09420',
                 '01736', '05222', '10835', '08539', '12832',
                 '01735', '01734', '04928']

data_val = [
    '092117L', '092117S', '052516', '032818', '041818', '022618',
    '080217', '061217', '110214', '040716', '083115', '071218', '040417'
]

data_test = [
    '100317L', '100317S', '120717', '090517L', '050318L', '051817',
    '062117', '102617', '092817L', '040218', '041017', '101317',
    '082517L', '041318L',
]

zoom_womb = ['ZM1', 'ZM2', 'ZM3', 'ZM4', 'ZM5', 'ZM6', 'ZM7', 'ZM8', 'ZM9', 'ZM10']


# ensure there is no duplicates across the splits
assert len(set(data_train).intersection(set(data_val))) == 0, 'There are duplicates between the training and validation sets.'
assert len(set(data_train).intersection(set(data_test))) == 0, 'There are duplicates between the training and testing sets.'
assert len(set(data_val).intersection(set(data_test))) == 0, 'There are duplicates between the validation and testing sets.'
print(f"Training: {len(data_train)} | Validation: {len(data_val)} | Testing: {len(data_test)}")


# Rotations
rots = [[], [(1, (0,1))], [(1, (1,2))], [(1, (2,0))], [(2, (0,1))], [(1, (0,1)), (1, (1,2))],
        [(1, (0,1)), (1, (2,0))], [(1, (1,2)), (1, (0,1))], [(2, (1,2))], [(1, (2,0)), (1, (1,2))],
        [(2, (2,0))], [(3, (0,1))], [(2, (0,1)), (1, (1,2))], [(2, (0,1)), (1, (2,0))],
        [(2, (1,2)), (1, (2,0))], [(1, (0,1)), (2, (1,2))], [(1, (0,1)), (2, (2,0))],
        [(1, (1,2)), (2, (0,1))], [(3, (1,2))], [(3, (2,0))], [(3, (0,1)), (1, (1,2))],
        [(2, (0,1)), (1, (1,2)), (1, (0,1))], [(3, (1,2)), (1, (2,0))], [(1, (0,1)), (3, (1,2))]]

# utility functions
def random_rot(*args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    else:
        args = list(args)
    rot = choice(rots)
    for k, axes in rot:
        # choose 90 or 180 degree rotation
        if random() < 0.5:
            #print('90 degree rotation')
            for i in range(len(args)):
                args[i] = np.rot90(args[i], k, (axes[0]+1, axes[1]+1)) if args[i] is not None else None
        else:
            #print('180 degree rotation')
            for i in range(len(args)):
                args[i] = np.rot90(np.rot90(args[i], k, (axes[1]+1, axes[0]+1)), k, (axes[1]+1, axes[0]+1)) if args[i] is not None else None
    return args

def random_rot_list(*args):
    # If the input is a single list, treat it as a sequence of arguments
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    # Choose a random rotation configuration
    rot = choice(rots)

    # Perform the rotation on each argument (which can be an image or array)
    for k, axes in rot:
        for i in range(len(args)):
            if args[i] is not None:
                args[i] = np.rot90(args[i], k, (axes[0] + 1, axes[1] + 1))
            else:
                args[i] = None
    
    return args

def read_nifti(nii_filename):
    data = nib.load(nii_filename)
    res = data.header.get_zooms()
    return np.squeeze(data.get_fdata().astype(np.float32)), res

# augmentation classes
class RandomDropout:
    def __init__(self, prob: float = 0.1, max_number: int = 1):
        self.prob = prob
    def forward(self, volume, labels):
        
        if np.random.rand() < self.prob:
            num = np.random.randint(1, self.max_number)
            for i in range(num):
                # zero out a 3x3 region in the volume centered around a random coordinate in labels
                x, y, z = labels[np.random.randint(0, len(labels))]
                volume[:, x-1:x+2, y-1:y+2, z-1:z+2] = 0
                
                
                
        return volume, labels

class NormalizeByPercentile(tio.Transform):
    """
    Normalize volume by dividing by the 99th percentile of non-zero values.
    
    Args:
        percentile: The percentile value to use for normalization (default: 99)
    """
    
    def __init__(self, percentile=99, scale=False):
        super().__init__()
        self.percentile = percentile
        self.scale = scale
    
    def apply_transform(self, subject):
        # Create a copy of the subject to avoid modifying the original
        transformed = subject.copy()

        # Access the data tensor
        data = transformed['raw_volume'].data
        
        # Calculate percentile of non-zero values for the first channel
        percentile_fac = np.percentile(data[0][data[0] > 0], self.percentile)
        
        if self.scale is True: # scale from 0 to 1
            # Normalize the data
            subject['raw_volume'].set_data(data / percentile_fac)
            
            # Scale the data
            subject['raw_volume'].set_data((data - data.min()) / (data.max() - data.min()))
        else:
            # Normalize the data
            subject['raw_volume'].set_data(data / percentile_fac)
        
        return subject
    
class Zoom(tio.Transform):
    """
    Zoom volume by a factor.
    
    Args:
        factor: The zoom factor to use for zooming (default: 1)
    """
    
    def __init__(self, zf):
        super().__init__()
        self.zf = zf
    def apply_transform(self, subject):
        # Create a copy of the subject to avoid modifying the original
        transformed = subject.copy()

        # Loop over all the volumes in the subject
        for image in subject.keys():
            subject[f'{image}'].set_data(zoom(transformed[f'{image}'].data, self.zf, order=1))

        return subject

class FinalDataset(torch.utils.data.Dataset):
    def __init__(self, stage, opts):
        # Set the self variables
        self.stage = stage
        self.opts  = opts
        self.subjs = sorted(globals()['data_' + stage])
        self.norm  = NormalizeByPercentile()
        self.initialize_augmentations()

        if self.opts.use_zoom_womb is True:
            zw = sorted(globals()['zoom_womb'])
            print(f'Length of the original dataset: {len(self.subjs)} | Length of {len(zw)}' )
            self.subjs.extend(zw)
            print(f'Length of the new dataset: {len(self.subjs)}')
        
    
        self.data = []

        # now load the subjects
        for dn, subj in enumerate(self.subjs):
            # check if subject is in zoom_womb
            if subj in globals()['zoom_womb']:
                zw = True
            else:
                zw = False

            # get the folder of the subject
            folder = os.path.join(opts.rawdata_path, subj)

            # get the joint labels
            label_file = os.path.join(opts.label_path, subj + '.mat')
            labels = sio.loadmat(label_file)['joint_coord'].astype(np.int32)
            labels = labels[:, [1, 0, 2]]  # convert to z, x, y

            # get the filenames
            niinames = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]

            for filename, label in zip(niinames, labels):
                self.data.append({'foldername': subj, 'filename': filename, 'label': label, 'sid': dn, 'zoomwomb': zw})
        print(f"Stage: {stage} | Total Volumes: {len(self.data)}")

    def initialize_augmentations(self):
        # Probability of applying any given augmentation
        p = 0.8
        # Define individual augmentations as a dictionary
        self.augmentations = {}

        self.augmentations['noise'] = (
            tio.RandomNoise(std=(0.02, 0.03), p=p)
            if self.opts.noise
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['spike'] = (
            tio.RandomSpike(num_spikes=(4), intensity=0.3, p=p)
            if self.opts.spike
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['bfield'] = (
            tio.RandomBiasField(coefficients=(0.1, 0.165), p=p)
            if self.opts.bfield
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['gamma'] = (
            tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)
            if self.opts.gamma
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['anisotropy'] = (
            tio.OneOf([
                tio.RandomAnisotropy(axes=(0,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(1,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(2,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 2), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(1, 2), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 2), p=p),
            ])
            if self.opts.anisotropy
            else tio.Lambda(lambda x: x)
        )
        # This augmentation is always applied
        self.augmentations['clamp'] = tio.Clamp(out_min=0)

    def apply_augmentations(self, subject, zoom_womb, label, stage):
        zf = 1
        augmentation_pipeline = tio.Compose([])

        # Apply custom augmentation for training if needed
        if stage == 'train' and self.opts.custom_augmentation is True and not zoom_womb:
            if random() < self.opts.augmentation_prob:
                if random() < 0.5 and self.opts.zoom > 0:
                        if random() < self.opts.zoom and zoom_womb is False:
                            zf      = uniform(0.55, 0.65)
                            zoomer  = Zoom(zf=zf)
                            subject = zoomer(subject)
                            augmentation_pipeline = tio.Compose([self.augmentations['noise'],
                                                                self.augmentations['clamp']])  # Reapply the clamp after zooming
                        else:
                            augmentation_pipeline = tio.Compose([self.augmentations['noise'],
                                                                 self.augmentations['spike'],
                                                                 tio.OneOf({self.augmentations['bfield'], self.augmentations['gamma']}),
                                                                 self.augmentations['clamp']])
                            
                else:
                    augmentation_pipeline = tio.Compose([self.augmentations['noise'],
                                                        tio.OneOf({self.augmentations['bfield'], self.augmentations['gamma']}),
                                                        self.augmentations['anisotropy'],
                                                        self.augmentations['clamp']])

        # Apply the composed augmentation pipeline
        transformed = augmentation_pipeline(subject)
        volume      = transformed['raw_volume'].data[0].numpy()
        joint_coord = label * zf
        return volume, joint_coord, zf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get data parameters
        data        = self.data[idx] # get data
        volume, r   = read_nifti(data['filename']) # read the volume
        r           = np.array(r) # get the resolution 
        r           = np.concatenate((r, [1])) # concatenate a 1 to the resolution
        zf          = 1 # add zoom factor
        zoom_womb   = data['zoomwomb'] # get if the data is zoom_womb
        time_index  = data['sid']

        # Start processing the data
        image       = tio.ScalarImage(tensor=volume[None, ...], affine=np.eye(4)*r)
        subject     = tio.Subject(raw_volume=image)
        factors     = np.ones((3,1))
        label       = data['label']
        if self.stage == 'finetune':
            subject = tio.Resample((3, 3, 3))(subject)
            label   = label + 1        
            factors = (3.0 / r[0], 3.0 / r[1], 3.0 / r[2])
            factors = np.expand_dims(factors, axis=1)
        subject     = self.norm(subject)

        v, jc, zf   = self.apply_augmentations(subject, zoom_womb, label, self.stage)
        v, o        = self.crop(v)
        v           = np.expand_dims(v, axis=0)
        jc          = jc * (1/factors)
        if self.stage == 'train' or self.stage == 'val':
            hmap = self.gen_hmap(jc, o, zf)
        
        if self.stage == 'train' or self.stage == 'val' or self.stage == 'finetune':
            hmap = self.gen_hmap(jc, o, zf)
            if self.stage == 'train' and self.opts.rot is True:
                v, hmap         = random_rot(v, hmap)
                v               = v.copy()
                hmap            = hmap.copy()
            
            return v, 0, hmap
        
        if self.stage == 'test':
            return v, jc, data['sid']

    # cropping function
    def crop(self, volume, origin=None):
        cs = self.opts.crop_size
        if self.stage == 'test':
            if self.opts.unet_type == 'small':
                depth = self.opts.depth + 1
            else:
                depth = self.opts.depth
            
            f = 2**depth
            pad_width = [(0, ceil(s/f)*f-s) for s in volume.shape]
            volume = np.pad(volume, pad_width, mode='constant')
            x_0 = y_0 = z_0 = 0
        else:
            if any(s < cs for s in volume.shape):
                pad_width = [(0, max(0, cs-s)) for s in volume.shape]
                volume = np.pad(volume, pad_width, mode='constant')
            if origin:
                x_0, y_0, z_0 = origin
            else:
                x_0, y_0, z_0 = randint(0, volume.shape[1] - cs), randint(0, volume.shape[0] - cs), randint(0, volume.shape[2] - cs) # close interval
            volume = volume[y_0:y_0+cs, x_0:x_0+cs, z_0:z_0+cs]
        return volume, (x_0, y_0, z_0)
    
    # heatmap generating function
    def gen_hmap(self, joint, origin, zf=1): # Added zf as an argument for zoom factor, may need to change this

        joint = np.around(joint).astype(np.int32)

        cs = self.opts.crop_size
        x_0, y_0, z_0 = origin
        y_range = np.reshape(np.arange(y_0+1, y_0+cs+1, dtype=np.float32), (1,-1,1,1))
        x_range = np.reshape(np.arange(x_0+1, x_0+cs+1, dtype=np.float32), (1,1,-1,1))
        z_range = np.reshape(np.arange(z_0+1, z_0+cs+1, dtype=np.float32), (1,1,1,-1))

        x_label, y_label, z_label = np.reshape(joint, (3,-1,1,1,1))
        dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
        dd = dx**2 + dy**2 + dz**2

        heatmap = self.opts.mag * (2.0 ** 3) / (2 ** 3) * np.exp(-0.5 / (self.opts.sigma * zf) ** 2 * dd)
        return heatmap.astype(np.float32, copy=False)

# Define the dataset class
"""class Dataset(torch.utils.data.Dataset):
    def __init__(self, stage, opts):
        # Set the self variables
        self.stage      = stage
        self.opts       = opts
        self.subjects   = sorted(globals()['data_' + stage])
        if self.opts.use_zoom_womb is True and self.stage == 'train':
            self.subjects.extend(sorted(globals()['zoom_womb']))

        self.data       = []
        
        # Load the subjects
        for dn, subject in enumerate(self.subjects):
            # Check if subject is in zoom_womb
            if subject in globals()['zoom_womb']:
                zw = True
            else:
                zw = False
            
            # Get the folder of the subject
            folder      = os.path.join(opts.rawdata_path, subject)
            
            # Get the joint labels
            label_file  = os.path.join(opts.label_path, subject + '.mat')
            labels      = sio.loadmat(label_file)['joint_coord'].astype(np.int32)

            # Get the filenames
            niinames    = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
            
            for filename, label in zip(niinames, labels):
                self.data.append({'foldername': subject, 'filename': filename, 'label': label, 'sid': dn, 'zm': zw})
        
        # Print the stage and data
        print(f"Stage: {stage} | Total Volumes: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):       
        # Get the data
        data        = self.data[idx]
        volume, _ = read_nifti(data['filename'])
        sid         = data['sid']
        zf          = 1
        zoom_womb   = data['zm']
        # data augmentation
        
        ## Put subject into TorchIO format
        subject = tio.Subject(raw_volume=tio.ScalarImage(tensor=volume[None, ...]))

        ## Determine normalization type
        if self.opts.norm_type == 'percentile':
            necessary = NormalizeByPercentile()
        elif self.opts.norm_type == 'percentile_window':
            necessary = tio.RescaleIntensity((0, 1), (1., 99.0))
        


        ## Determine the augmentations
        if self.stage == 'train' and self.opts.custom_augmentation is True and zoom_womb is False:
            if random() < self.opts.augmentation_prob:
                p = 0.8
                # if random() < 0.5: then apply augmentations, no anistropy and then zoom
                if random() < 0.5:

                    if self.opts.zoom > 0:
                        
                        if random() < self.opts.zoom:
                            # Sample a random zoom factor
                            zf              = uniform(0.55, 0.65)
                            zoomer          = Zoom(zf=zf)
                            
                            # Apply normalization and zoom
                            subject         = necessary(subject)
                            subject         = zoomer(subject)

                            # Define some augmentations
                            augmentations   = tio.Compose([
                                                tio.RandomNoise(std=(0.02, 0.03), p=p),
                                                tio.Clamp(out_min=0),
                                                ])
                        
                            # Apply the augmentations
                            transformed     = augmentations(subject)
                        
                            # Convert the subject data to numpy
                            volume          = transformed['raw_volume'].data[0].numpy()
                            joint_coord     = data['label'] * zf
                        else:
                            # Def the augmentations // no zoom
                            augmentations   = tio.Compose([
                                necessary,
                                tio.RandomNoise(std=(0.02, 0.03), p=p),
                                tio.RandomSpike(num_spikes=(4), intensity=0.3, p=p),
                                tio.OneOf({tio.RandomBiasField(coefficients=(0.1, 0.165), p=p),tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)}),
                                tio.Clamp(out_min=0),
                            ])
                            
                            # Apply the augmentations
                            transformed         = augmentations(subject)
                        
                            # Convert the subject data to numpy
                            volume              = transformed['raw_volume'].data[0].numpy()
                            zf                  = 1
                            joint_coord         = data['label']
                    
                else:
                    random_anisotropy = tio.OneOf({tio.RandomAnisotropy(axes=(0), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(1), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(2), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(0, 2), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(1, 2), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5,2), p=p),
                        })
                    
                    ## Construct the transformations
                    augmentations = tio.Compose([
                        necessary,
                        tio.RandomNoise(std=(0.02, 0.03), p=p),
                        tio.OneOf({tio.RandomBiasField(coefficients=(0.1, 0.165), p=p),tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)}),
                        random_anisotropy,
                        tio.Clamp(out_min=0),
                    ])
                    
                    ## Choose whether to augment or not
                    transforms = augmentations
                    
                    ## Apply the transformations
                    transformed = transforms(subject)
                    
                    # Convert the subject data to numpy
                    volume      = transformed['raw_volume'].data[0].numpy()
                    joint_coord = data['label']
                    zf          = 1
            # No augmentation
            else:
                # Apply normalization
                transformed = necessary(subject)
                volume      = transformed['raw_volume'].data[0].numpy()
                joint_coord = data['label']
            
        else:
            # If we are using the zoom_womb dataset
            if zoom_womb is True and self.stage == 'train':
                
                transformed = necessary(subject)
                volume      = transformed['raw_volume'].data[0].numpy()
                joint_coord = (data['label'] * zf)
                zf          = 1
            
            # If we are using the baseline model
            elif self.opts.baseline is True and self.stage == 'train' and random() < self.opts.zoom:
                if self.opts.zoom <= 1:
                    zf = uniform(1/self.opts.zoom_factor, self.opts.zoom_factor)
                else:
                    zf = self.opts.zoom_factor
                zoomer          = Zoom(zf=zf)
                transformed     = zoomer(subject)
                transformed     = necessary(subject)
                volume          = transformed['raw_volume'].data[0].numpy()
                joint_coord     = data['label'] * zf
                zf              = 1 # because Junshen doesn't scale sigma
            
            # If we are doing nothing to the data // validation or test
            else:
                transformed = necessary(subject)
                volume      = transformed['raw_volume'].data[0].numpy()
                joint_coord = data['label']
                zf          = 1

        # Crop the image
        volume, origin = self.crop(volume) 
        
        # If temporal training, get the previous volume            
        volume_prev = np.zeros_like(volume)
        
        # Get the heatmaps if the stage is train or val
        if self.stage == 'train' or self.stage == 'val':
            heatmap = self.gen_hmap(joint_coord, origin, zf)
        
        volume      = np.expand_dims(volume, axis=0)
        if volume_prev is not None:
            volume_prev = np.zeros_like(volume)           
        
        # Apply rotations if the stage is train
        if self.stage == 'train':
            if self.opts.rot is True:
                volume, volume_prev, heatmap = random_rot(volume, volume_prev, heatmap)
                volume          = volume.copy()
                heatmap         = heatmap.copy()
                volume_prev     = volume_prev.copy()
                
            if self.opts.junshen_scale is not None:
                volume = volume ** uniform(
                1 - self.opts.junshen_scale, 1 + self.opts.junshen_scale
            )
        
        if self.stage == 'test':
            return volume, joint_coord, data['sid']
        else:
            return volume, (0 if volume_prev is None else volume_prev), heatmap
        
    # cropping function
    def crop(self, volume, origin=None):
        cs = self.opts.crop_size
        if self.stage == 'test':
            if self.opts.unet_type == 'small':
                depth = self.opts.depth + 1
            else:
                depth = self.opts.depth
            
            f = 2**depth
            pad_width = [(0, ceil(s/f)*f-s) for s in volume.shape]
            volume = np.pad(volume, pad_width, mode='constant')
            x_0 = y_0 = z_0 = 0
        else:
            if any(s < cs for s in volume.shape):
                pad_width = [(0, max(0, cs-s)) for s in volume.shape]
                volume = np.pad(volume, pad_width, mode='constant')
            if origin:
                x_0, y_0, z_0 = origin
            else:
                x_0, y_0, z_0 = randint(0, volume.shape[1] - cs), randint(0, volume.shape[0] - cs), randint(0, volume.shape[2] - cs) # close interval
            volume = volume[y_0:y_0+cs, x_0:x_0+cs, z_0:z_0+cs]
        return volume, (x_0, y_0, z_0)
    
    # heatmap generating function
    def gen_hmap(self, joint, origin, zf=1): # Added zf as an argument for zoom factor, may need to change this

        joint = np.around(joint).astype(np.int32)

        cs = self.opts.crop_size
        x_0, y_0, z_0 = origin
        y_range = np.reshape(np.arange(y_0+1, y_0+cs+1, dtype=np.float32), (1,-1,1,1))
        x_range = np.reshape(np.arange(x_0+1, x_0+cs+1, dtype=np.float32), (1,1,-1,1))
        z_range = np.reshape(np.arange(z_0+1, z_0+cs+1, dtype=np.float32), (1,1,1,-1))

        x_label, y_label, z_label = np.reshape(joint, (3,-1,1,1,1))
        dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
        dd = dx**2 + dy**2 + dz**2

        heatmap = self.opts.mag * (2.0 ** 3) / (2 ** 3) * np.exp(-0.5 / (self.opts.sigma * zf) ** 2 * dd)
        return heatmap.astype(np.float32, copy=False)

    # 
    def augmentations(self, subject, zoom_womb, data, stage):
        # probability to apply any given augmentation
        p   = 0.8
        # zoom-factor placeholder
        zf  = 1

        # get the list of augmentations
        noise   = tio.Compose([tio.RandomNoise(std=(0.02, 0.03), p=p)] if self.opts.noise else [])
        spike   = tio.Compose([tio.RandomSpike(num_spikes=(4), intensity=0.3, p=p)] if self.opts.spike else [])
        bfield  = tio.Compose([tio.RandomBiasField(coefficients=(0.1, 0.165), p=p)] if self.opts.bfield else [])
        gamma   = tio.Compose([tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)] if self.opts.gamma else [])
        aniso   = tio.Compose([tio.OneOf([
                        tio.RandomAnisotropy(axes=(0), downsampling=(1.5, 2), p=p),
                        tio.RandomAnisotropy(axes=(1), downsampling=(1.5, 2), p=p),
                        tio.RandomAnisotropy(axes=(2), downsampling=(1.5, 2), p=p),
                        tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.5, 2), p=p),
                        tio.RandomAnisotropy(axes=(0, 2), downsampling=(1.5, 2), p=p),
                        tio.RandomAnisotropy(axes=(1, 2), downsampling=(1.5, 2), p=p),
                        tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 2), p=p),
                    ])] if self.opts.anisotropy else [])

        if stage == 'train' and self.opts.custom_augmentation and not zoom_womb:
            if random() < self.opts.augmentation_prob:
                if random() < 0.5:
                    if self.opts.zoom > 0 and random() < self.opts.zoom:
                        zf = uniform(0.55, 0.65)
                        zoomer = Zoom(zf=zf)
                        subject = zoomer(subject)
                        
                        augmentations = tio.Compose([
                            noise,
                            tio.Clamp(out_min=0),
                        ])
                        transformed = augmentations(subject)
                        volume = transformed['raw_volume'].data[0].numpy()
                        joint_coord = data['label'] * zf
                    else:
                        augmentations = tio.Compose([
                            noise,
                            spike,
                            tio.OneOf({
                                bfield,
                                gamma,
                            }),
                            tio.Clamp(out_min=0),
                        ])
                        transformed = augmentations(subject)
                        volume = transformed['raw_volume'].data[0].numpy()
                        joint_coord = data['label']
                else:
                    
                    augmentations = tio.Compose([
                        noise,
                        tio.OneOf({
                            bfield,
                            gamma
                        }),
                        aniso,
                        tio.Clamp(out_min=0),
                    ])
                    transformed = augmentations(subject)
                    volume = transformed['raw_volume'].data[0].numpy()
                    joint_coord = data['label']
            else:
                volume      = transformed['raw_volume'].data[0].numpy()
                joint_coord = data['label']
        return volume, joint_coord, zf
"""

# Define the ablation class // probably the one I will end up with
class Dataset(torch.utils.data.Dataset):
    def __init__(self, stage, opts):
        #############
        ## STAGING ##
        #############

        self.stage      = stage
        self.opts       = opts
        self.subjects   = sorted(globals()['data_' + stage])
        self.norm       = NormalizeByPercentileTemporal(opts)

        ## Determine whether to use `zoom womb` or `fetal inpainting`... they are the same thing.
        if self.opts.use_zoom_womb is True and self.stage == 'train':
            self.subjects.extend(sorted(globals()['zoom_womb']))

        # Load subjects and associated filenames/labels
        self.data = []
        for dn, subject in enumerate(self.subjects):
            zw          = subject in globals()['zoom_womb']
            folder      = os.path.join(opts.rawdata_path, subject)
            label_file  = os.path.join(opts.label_path, subject + '.mat')
            labels      = sio.loadmat(label_file)['joint_coord'].astype(np.int32)
            niinames    = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
            for filename, label in zip(niinames, labels):
                self.data.append({'foldername': subject, 'filename': filename, 'label': label, 'sid': dn, 'zm': zw})
        
        self.initialize_augmentations()
        print(f"Stage: {stage} | Total Volumes: {len(self.data)}")
    
    def initialize_augmentations(self):
        p = 0.8
        self.augmentations = {}
        self.augmentations['noise'] = (
            tio.RandomNoise(std=(0.02, 0.03), p=p)
            if self.opts.noise
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['spike'] = (
            tio.RandomSpike(num_spikes=4, intensity=0.3, p=p)
            if self.opts.spike
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['bfield'] = (
            tio.RandomBiasField(coefficients=(0.1, 0.165), p=p)
            if self.opts.bfield
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['gamma'] = (
            tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)
            if self.opts.gamma
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['anisotropy'] = (
            tio.OneOf([
                tio.RandomAnisotropy(axes=(0,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(1,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(2,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 2), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(1, 2), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 2), p=p),
            ])
            if self.opts.anisotropy
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['clamp'] = tio.Clamp(out_min=0)

    def apply_augmentations(self, subject, label, zoom_womb, stage):
        # Default zoom factor is 1 (no zoom)
        zoom_factor          = 1
        shared_transforms    = []
        per_image_transforms = []

        if stage == 'train' and self.opts.custom_augmentation and not zoom_womb:
            if random() < self.opts.augmentation_prob:
                if random() < 0.5:
                    # Optionally apply zoom as a shared transform
                    if self.opts.zoom > 0 and random() < self.opts.zoom:
                        zoom_factor = uniform(0.55, 0.65)
                        zoomer = Zoom(zf=zoom_factor)
                        shared_transforms.append(zoomer)
                        per_image_transforms.append(self.augmentations['noise'])
                        per_image_transforms.append(self.augmentations['clamp'])
                    else:
                        per_image_transforms.extend([
                            self.augmentations['noise'],
                            self.augmentations['spike'],
                            tio.OneOf([self.augmentations['bfield'], self.augmentations['gamma']])
                        ])
                        per_image_transforms.append(self.augmentations['clamp'])
                else:
                    # Apply anisotropy as a shared transform
                    shared_transforms.append(self.augmentations['anisotropy'])
                    per_image_transforms.append(self.augmentations['noise'])
                    per_image_transforms.append(self.augmentations['clamp'])
        else:
            shared_transforms.append(self.augmentations['clamp'])

        # Compose and apply the shared (subject-level) transforms
        shared_pipeline = tio.Compose(shared_transforms)
        subject = shared_pipeline(subject)
        
        # Then apply the per-image transforms individually
        per_image_pipeline = tio.Compose(per_image_transforms)
        for image_name in subject.get_images_dict():
            #transformed_image = shared_pipeline(subject[image_name])
            transformed_image = per_image_pipeline(subject[image_name])
            subject[image_name].set_data(transformed_image.data)

        # Update joint coordinates using the zoom factor
        joint_coord = label * zoom_factor
        return subject, joint_coord, zoom_factor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data        = self.data[idx]
        volume, _   = read_nifti(data['filename'])
        sid         = data['sid']
        zoom_womb   = data['zm']

        ## Load the subject using TorchIO ##
        subject_dict                = {}
        subject_dict["raw_volume"]  = tio.ScalarImage(tensor=torch.tensor(volume[None, ...]))
        subject                     = tio.Subject(**subject_dict)

        # Normalize the subject
        subject                     = self.norm(subject)
        
        # Apply the shared and per-image augmentation pipeline only once.
        subject, joint_coord, zoom_factor = self.apply_augmentations(subject, data['label'], zoom_womb, self.stage)
        
        # Extract augmented volumes
        aug_volume          = subject['raw_volume'].data[0].numpy()
        aug_volume, origin  = self.crop(aug_volume)

        # Generate heatmap if in train and val
        if self.stage in ['train', 'val']:
            heatmap = self.gen_hmap(joint_coord, origin, zoom_factor)
                
        # Ensure primary volume has a channel dimension
        aug_volume = np.expand_dims(aug_volume, axis=0)

        if self.stage == 'train':
            #
            if self.opts.rot:
                aug_volume, heatmap = random_rot(aug_volume, heatmap)
                aug_volume          = aug_volume.copy()
                heatmap             = heatmap.copy()

            # Junshen's gamma augmentations
            if self.opts.junshen_scale is not None:
                scale_factor = uniform(1 - self.opts.junshen_scale, 1 + self.opts.junshen_scale)
                aug_volume   = aug_volume ** scale_factor
        
        ## Returning for single frame training ##
        if self.stage in ['train', 'val']:
            return aug_volume, heatmap
        if self.stage == 'test':
            return aug_volume, joint_coord, data['sid']

    def crop(self, volume, origin=None):
        cs = self.opts.crop_size
        if self.stage == 'test':
            f           = 2 ** self.opts.depth
            pad_width   = [(0, ceil(s / f) * f - s) for s in volume.shape]
            volume      = np.pad(volume, pad_width, mode='constant')
            x_0 = y_0 = z_0 = 0
        else:
            if any(s < cs for s in volume.shape):
                pad_width = [(0, max(0, cs - s)) for s in volume.shape]
                volume = np.pad(volume, pad_width, mode='constant')
            if origin:
                x_0, y_0, z_0 = origin
            else:
                x_0 = randint(0, volume.shape[1] - cs)
                y_0 = randint(0, volume.shape[0] - cs)
                z_0 = randint(0, volume.shape[2] - cs)
            volume = volume[y_0:y_0+cs, x_0:x_0+cs, z_0:z_0+cs]
        return volume, (x_0, y_0, z_0)
    
    def gen_hmap(self, joint, origin, zoom_factor=1):
        # Round the joint coordinates
        joint           = np.around(joint).astype(np.int32)

        # Define the crop size
        cs              = self.opts.crop_size
        
        # Set the origin
        x_0, y_0, z_0   = origin

        # Define the Gaussian parameters
        y_range         = np.reshape(np.arange(y_0+1, y_0+cs+1, dtype=np.float32), (1, -1, 1, 1))
        x_range         = np.reshape(np.arange(x_0+1, x_0+cs+1, dtype=np.float32), (1, 1, -1, 1))
        z_range         = np.reshape(np.arange(z_0+1, z_0+cs+1, dtype=np.float32), (1, 1, 1, -1))
        x_label, y_label, z_label = np.reshape(joint, (3, -1, 1, 1, 1))
        dx, dy, dz      = x_range - x_label, y_range - y_label, z_range - z_label
        dd = dx**2 + dy**2 + dz**2

        # Create the Gaussian
        heatmap = self.opts.mag * np.exp(-0.5 / (self.opts.sigma * zoom_factor) ** 2 * dd)
        return heatmap.astype(np.float32, copy=False)



# This is temporal do not use
class DSET(torch.utils.data.Dataset):
    def __init__(self, stage, opts):
        #############
        ## STAGING ##
        #############

        self.stage      = stage
        self.opts       = opts
        self.subjects   = sorted(globals()['data_' + stage])
        self.norm       = NormalizeByPercentileTemporal(opts)

        ## Determine whether to use `zoom womb` or `fetal inpainting`... they are the same thing.
        if self.opts.use_zoom_womb is True and self.stage == 'train':
            self.subjects.extend(sorted(globals()['zoom_womb']))

        # Load subjects and associated filenames/labels
        self.data = []
        for dn, subject in enumerate(self.subjects):
            zw          = subject in globals()['zoom_womb']
            folder      = os.path.join(opts.rawdata_path, subject)
            label_file  = os.path.join(opts.label_path, subject + '.mat')
            labels      = sio.loadmat(label_file)['joint_coord'].astype(np.int32)
            niinames    = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
            for filename, label in zip(niinames, labels):
                self.data.append({'foldername': subject, 'filename': filename, 'label': label, 'sid': dn, 'zm': zw})
        
        self.initialize_augmentations()
        print(f"Stage: {stage} | Total Volumes: {len(self.data)}")
    
    def initialize_augmentations(self):
        p = 0.8
        self.augmentations = {}
        self.augmentations['noise'] = (
            tio.RandomNoise(std=(0.02, 0.03), p=p)
            if self.opts.noise
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['spike'] = (
            tio.RandomSpike(num_spikes=4, intensity=0.3, p=p)
            if self.opts.spike
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['bfield'] = (
            tio.RandomBiasField(coefficients=(0.1, 0.165), p=p)
            if self.opts.bfield
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['gamma'] = (
            tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)
            if self.opts.gamma
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['anisotropy'] = (
            tio.OneOf([
                tio.RandomAnisotropy(axes=(0,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(1,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(2,), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 2), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(1, 2), downsampling=(1.5, 2), p=p),
                tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 2), p=p),
            ])
            if self.opts.anisotropy
            else tio.Lambda(lambda x: x)
        )
        self.augmentations['clamp'] = tio.Clamp(out_min=0)

    def apply_augmentations(self, subject, label, zoom_womb, stage):
        # Default zoom factor is 1 (no zoom)
        zoom_factor = 1

        shared_transforms = []
        per_image_transforms = []

        if stage == 'train' and self.opts.custom_augmentation and not zoom_womb:
            if random() < self.opts.augmentation_prob:
                if random() < 0.5:
                    # Optionally apply zoom as a shared transform
                    if self.opts.zoom > 0 and random() < self.opts.zoom:
                        zoom_factor = uniform(0.55, 0.65)
                        zoomer = Zoom(zf=zoom_factor)
                        shared_transforms.append(zoomer)
                        per_image_transforms.append(self.augmentations['noise'])
                        per_image_transforms.append(self.augmentations['clamp'])
                    else:
                        per_image_transforms.extend([
                            self.augmentations['noise'],
                            self.augmentations['spike'],
                            tio.OneOf([self.augmentations['bfield'], self.augmentations['gamma']])
                        ])
                        per_image_transforms.append(self.augmentations['clamp'])
                else:
                    # Apply anisotropy as a shared transform
                    shared_transforms.append(self.augmentations['anisotropy'])
                    per_image_transforms.append(self.augmentations['noise'])
                    per_image_transforms.append(self.augmentations['clamp'])
        else:
            shared_transforms.append(self.augmentations['clamp'])

        # Compose and apply the shared (subject-level) transforms
        shared_pipeline = tio.Compose(shared_transforms)
        subject = shared_pipeline(subject)
        
        # Then apply the per-image transforms individually
        per_image_pipeline = tio.Compose(per_image_transforms)
        for image_name in subject.get_images_dict():
            #transformed_image = shared_pipeline(subject[image_name])
            transformed_image = per_image_pipeline(subject[image_name])
            subject[image_name].set_data(transformed_image.data)

        # Update joint coordinates using the zoom factor
        joint_coord = label * zoom_factor
        return subject, joint_coord, zoom_factor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data        = self.data[idx]
        volume, _   = read_nifti(data['filename'])
        sid         = data['sid']
        zoom_womb   = data['zm']

        # For temporal training, get neighboring frame if available
        neighbor_volumes    = []
        neighbor_labels     = []
        left, right         = None, None
        if self.opts.temporal in [2, 3, 5]:
            # check if there are neighboring frames
            if 0 <= idx - (self.opts.temporal - 1) < len(self.data) and \
            0 <= idx + (self.opts.temporal - 1) < len(self.data) and \
            self.data[idx - (self.opts.temporal - 1)]['sid'] == sid and \
            self.data[idx + (self.opts.temporal - 1)]['sid'] == sid:

                # Collect the neighboring frames
                if self.opts.temporal > 2:
                    half_window = self.opts.temporal // 2
                    for ii in range(max(0, idx - half_window), min(len(self.data), idx + half_window + 1)):
                        if idx == ii:
                            continue
                        neighbor_volume, _ = read_nifti(self.data[ii]['filename'])
                        neighbor_volumes.append(neighbor_volume)
                        neighbor_labels.append(self.data[ii]['label'])
                
                # choose a random neighbor if it is temporal window == 2
                else:
                    candidates = [i for i in [idx - 1, idx + 1] if 0 <= i < len(self.data) and self.data[i]['sid'] == sid]
                    if candidates:
                        random_neighbor = choice(candidates)
                        neighbor_volume, _ = read_nifti(self.data[random_neighbor]['filename'])
                        neighbor_volumes.append(neighbor_volume)
                        neighbor_labels.append(self.data[random_neighbor]['label'])
            elif 0 <= idx - (self.opts.temporal - 1) < len(self.data) and self.data[idx - (self.opts.temporal - 1)]['sid'] == sid:
                # Left-side case
                left = True
                for ii in range(max(0, idx - self.opts.temporal + 1), idx):
                    if idx == ii:
                        continue
                    neighbor_volume, _ = read_nifti(self.data[ii]['filename'])
                    neighbor_volumes.append(neighbor_volume)
                    neighbor_labels.append(self.data[ii]['label'])

            elif 0 <= idx + (self.opts.temporal - 1) < len(self.data) and self.data[idx + (self.opts.temporal - 1)]['sid'] == sid:
                # Right-side case
                right = True
                for ii in range(idx + 1, min(len(self.data), idx + self.opts.temporal)):
                    if idx == ii:
                        continue
                    neighbor_volume, _ = read_nifti(self.data[ii]['filename'])
                    neighbor_volumes.append(neighbor_volume)
                    neighbor_labels.append(self.data[ii]['label'])
        elif self.opts.temporal > 5:
            if 0 <= idx - (self.opts.temporal - 1) < len(self.data) and self.data[idx - (self.opts.temporal - 1)]['sid'] == sid:
                # Left-side case
                left = True
                for ii in range(max(0, idx - self.opts.temporal + 1), idx):
                    if idx == ii:
                        continue
                    neighbor_volume, _ = read_nifti(self.data[ii]['filename'])
                    neighbor_volumes.append(neighbor_volume)
                    neighbor_labels.append(self.data[ii]['label'])

            elif 0 <= idx + (self.opts.temporal - 1) < len(self.data) and self.data[idx + (self.opts.temporal - 1)]['sid'] == sid:
                # Right-side case
                right = True
                for ii in range(idx + 1, min(len(self.data), idx + self.opts.temporal)):
                    if idx == ii:
                        continue
                    neighbor_volume, _ = read_nifti(self.data[ii]['filename'])
                    neighbor_volumes.append(neighbor_volume)
                    neighbor_labels.append(self.data[ii]['label'])

        ## Load the subject using TorchIO ##
        subject_dict = {}
        subject_dict["raw_volume"] = tio.ScalarImage(tensor=torch.tensor(volume[None, ...]))
        if self.opts.temporal > 1 and neighbor_volumes:
            for i, vol in enumerate(neighbor_volumes):
                subject_dict[f"neighbor_volume_{i}"] = tio.ScalarImage(tensor=torch.tensor(vol[None, ...]))
        subject = tio.Subject(**subject_dict)

        # Normalize the subject
        subject = self.norm(subject)
        
        # Apply the shared and per-image augmentation pipeline only once.
        subject, joint_coord, zoom_factor = self.apply_augmentations(subject, data['label'], zoom_womb, self.stage)
        
        # Extract augmented volumes
        aug_volume          = subject['raw_volume'].data[0].numpy()
        neighbor_aug_list   = []
        if self.opts.temporal > 1:
            for key in subject.keys():
                # Only process keys that start with "neighbor_volume"
                if key.startswith("neighbor_volume"):
                    neighbor_img = subject[f'{key}'].data[0].numpy()
                    neighbor_aug_list.append(neighbor_img)
        
        if not neighbor_aug_list:
            aug_volume_neighbor = np.zeros_like(aug_volume)
            aug_volume, origin = self.crop(aug_volume)
        else:
            aug_volume, origin = self.crop(aug_volume)
            cropped_neighbors = []

            for neighbor in neighbor_aug_list:
                cropped, _ = self.crop(neighbor, origin)
                cropped_neighbors.append(cropped)

        # If a neighbor label is provided (or computed earlier), adjust it.
        if self.stage in ['train', 'val']:
            heatmap = self.gen_hmap(joint_coord, origin, zoom_factor)
            
            neighbor_heatmaps = []
            if self.opts.temporal > 1:
                # Loop over each neighbor label in your list 'neighbor_labels'
                for lbl in neighbor_labels:
                    adjusted_lbl = lbl * zoom_factor
                    hm = self.gen_hmap(adjusted_lbl, origin, zoom_factor)
                    neighbor_heatmaps.append(hm)
            else:
                neighbor_heatmaps = np.zeros_like(heatmap)
                
        # Ensure primary volume has a channel dimension
        aug_volume = np.expand_dims(aug_volume, axis=0)

        # Ensure neighbor volumes have a channel dimension
        if self.opts.temporal > 1:
            aug_volumes_neighbor = [np.expand_dims(vol, axis=0) for vol in cropped_neighbors]

        if self.stage == 'train':
            if self.opts.rot:
                if self.opts.temporal > 1:
                    # Apply rotation to all volumes and heatmaps
                    inputs = [aug_volume] + aug_volumes_neighbor + [heatmap] + neighbor_heatmaps
                    outputs = random_rot(inputs)
                    
                    # Extract transformed data
                    aug_volume = outputs[0].copy()
                    aug_volumes_neighbor = [out.copy() for out in outputs[1:len(aug_volumes_neighbor) + 1]]
                    heatmap = outputs[len(aug_volumes_neighbor) + 1].copy()
                    neighbor_heatmaps = [out.copy() for out in outputs[len(aug_volumes_neighbor) + 2:]]
                else:
                    aug_volume, heatmap = random_rot(aug_volume, heatmap)
                    volume = aug_volume.copy()
                    hmap = heatmap.copy()


            if self.opts.junshen_scale is not None:
                scale_factor = uniform(1 - self.opts.junshen_scale, 1 + self.opts.junshen_scale)
                aug_volume = aug_volume ** scale_factor
                aug_volumes_neighbor = [vol ** scale_factor for vol in aug_volumes_neighbor]
        

        ## Returning for single frame training ##
        if self.stage == 'train' and self.opts.temporal < 2:
            return aug_volume, heatmap
        if self.stage == 'val' and self.opts.temporal < 2:
            return aug_volume, heatmap
        if self.stage == 'test' and self.opts.temporal < 2:
            return aug_volume, joint_coord, data['sid']

        # Prepare concatenation of volumes and heatmaps
        all_volumes     = aug_volumes_neighbor  # List of neighboring volumes
        all_heatmaps    = neighbor_heatmaps  # List of neighboring heatmaps
        mid_index       = len(all_volumes) // 2  # Middle position
        if self.opts.temporal == 2:
            all_volumes.append(aug_volume)  # Append at the end
            all_heatmaps.append(heatmap)
        elif left == None and right == None:
            all_volumes.insert(mid_index, aug_volume)  # Insert in the middle
            all_heatmaps.insert(mid_index, heatmap)
        elif left is not None:
            all_volumes.insert(-1, aug_volume)  # Insert in the middle
            all_heatmaps.insert(-1, heatmap)
        elif right is not None:
            all_volumes.insert(0, aug_volume)  # Insert in the middle
            all_heatmaps.insert(0, heatmap)

        # Convert lists to arrays
        all_volumes = np.stack(all_volumes, axis=0)
        all_heatmaps = np.stack(all_heatmaps, axis=0)

        if not self.opts.tsm and not self.opts.four_dim:
            all_volumes     = all_volumes.reshape(-1, *all_volumes.shape[2:])
            all_heatmaps    = all_heatmaps.reshape(-1, *all_heatmaps.shape[2:])
        elif self.opts.four_dim:
            all_volumes     = rearrange(all_volumes, "T C H W D -> C T H W D")
            all_heatmaps    = rearrange(all_heatmaps, "T C H W D -> C T H W D")
        #print(f"{all_volumes.shape}, {all_heatmaps.shape}")
        if self.opts.middle_prediction:
            return all_volumes, all_heatmaps[:, 0]
        else:
            return all_volumes, all_heatmaps

    def crop(self, volume, origin=None):
        cs = self.opts.crop_size
        if self.stage == 'test':
            depth = self.opts.depth + 1 if self.opts.unet_type == 'small' else self.opts.depth
            f = 2 ** depth
            pad_width = [(0, ceil(s / f) * f - s) for s in volume.shape]
            volume = np.pad(volume, pad_width, mode='constant')
            x_0 = y_0 = z_0 = 0
        else:
            if any(s < cs for s in volume.shape):
                pad_width = [(0, max(0, cs - s)) for s in volume.shape]
                volume = np.pad(volume, pad_width, mode='constant')
            if origin:
                x_0, y_0, z_0 = origin
            else:
                x_0 = randint(0, volume.shape[1] - cs)
                y_0 = randint(0, volume.shape[0] - cs)
                z_0 = randint(0, volume.shape[2] - cs)
            volume = volume[y_0:y_0+cs, x_0:x_0+cs, z_0:z_0+cs]
        return volume, (x_0, y_0, z_0)
    
    def gen_hmap(self, joint, origin, zoom_factor=1):
        joint = np.around(joint).astype(np.int32)
        cs = self.opts.crop_size
        x_0, y_0, z_0 = origin
        y_range = np.reshape(np.arange(y_0+1, y_0+cs+1, dtype=np.float32), (1, -1, 1, 1))
        x_range = np.reshape(np.arange(x_0+1, x_0+cs+1, dtype=np.float32), (1, 1, -1, 1))
        z_range = np.reshape(np.arange(z_0+1, z_0+cs+1, dtype=np.float32), (1, 1, 1, -1))
        x_label, y_label, z_label = np.reshape(joint, (3, -1, 1, 1, 1))
        dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
        dd = dx**2 + dy**2 + dz**2
        heatmap = self.opts.mag * np.exp(-0.5 / (self.opts.sigma * zoom_factor) ** 2 * dd)
        return heatmap.astype(np.float32, copy=False)


# Define the zoomwomb dataset class // old do not use
class ZoomWombDataset(torch.utils.data.Dataset):
    def __init__(self, stage, opts):
        self.stage = stage
        self.opts = opts
        self.subjects = sorted(globals()['data_' + stage])
        
        # Separate regular and zoom_womb subjects
        self.zoom_womb_subjects = []
        if self.opts.use_zoom_womb is True and self.stage == 'train':
            self.zoom_womb_subjects = sorted(globals()['zoom_womb'])

        self.data = []
        self.zoom_womb_data = []  # Separate list for zoom_womb data
        
        # Load regular subjects
        for dn, subject in enumerate(self.subjects):
            # Check if subject is in zoom_womb
            zw = False
            
            folder = os.path.join(opts.rawdata_path, subject)
            label_file = os.path.join(opts.label_path, subject + '.mat')
            labels = sio.loadmat(label_file)['joint_coord'].astype(np.int32)
            niinames = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
            
            for filename, label in zip(niinames, labels):
                self.data.append({'foldername': subject, 'filename': filename, 'label': label, 'sid': dn, 'zm': zw})

        # Load zoom_womb subjects
        if self.zoom_womb_subjects:
            for dn, subject in enumerate(self.zoom_womb_subjects, start=len(self.subjects)):
                zw = True
                folder = os.path.join(opts.rawdata_path, subject)
                label_file = os.path.join(opts.label_path, subject + '.mat')
                labels = sio.loadmat(label_file)['joint_coord'].astype(np.int32)
                niinames = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.nii.gz')]
                
                for filename, label in zip(niinames, labels):
                    self.zoom_womb_data.append({'foldername': subject, 'filename': filename, 'label': label, 'sid': dn, 'zm': zw})

        # Calculate sampling weights for balanced sampling
        if self.stage == 'train' and self.zoom_womb_subjects:
            self.regular_size = len(self.data)
            self.zoom_womb_size = len(self.zoom_womb_data)
            
            # Determine sampling ratio (adjust these values based on your needs)
            target_regular = opts.dataset_size * 0.85  # 85% regular data
            target_zoom_womb = opts.dataset_size * 0.15  # 15% zoom_womb data
            
            self.regular_indices = list(range(self.regular_size))
            self.zoom_womb_indices = list(range(self.zoom_womb_size))
            
            # Calculate sampling weights
            self.regular_weights = [target_regular / self.regular_size] * self.regular_size
            self.zoom_womb_weights = [target_zoom_womb / self.zoom_womb_size] * self.zoom_womb_size
            
            print(f"Regular volumes: {self.regular_size}, Zoom womb volumes: {self.zoom_womb_size}")
            print(f"Sampling weights - Regular: {self.regular_weights[0]:.4f}, Zoom womb: {self.zoom_womb_weights[0]:.4f}")

        print(f"Stage: {stage} | Total Regular Volumes: {len(self.data)} | Total Zoom Womb Volumes: {len(self.zoom_womb_data)}")
    
    def create_subset(self, size=8000):
        """
        Creates a balanced subset of the dataset with stratified sampling
        """
        if not self.zoom_womb_subjects or self.stage != 'train':
            return torch.utils.data.Subset(self, torch.randperm(len(self))[:size])
        
        # Perform stratified sampling
        n_regular = int(size * 0.7)  # 70% regular data
        n_zoom_womb = size - n_regular  # 30% zoom_womb data
        
        # Sample indices with replacement if needed
        regular_indices = np.random.choice(
            self.regular_indices,
            size=n_regular,
            replace=len(self.regular_indices) < n_regular
        )
        
        zoom_womb_indices = np.random.choice(
            self.zoom_womb_indices,
            size=n_zoom_womb,
            replace=len(self.zoom_womb_indices) < n_zoom_womb
        )
        
        # Convert zoom_womb indices to dataset indices
        zoom_womb_indices = [i + self.regular_size for i in zoom_womb_indices]
        
        # Combine and shuffle indices
        combined_indices = np.concatenate([regular_indices, zoom_womb_indices])
        np.random.shuffle(combined_indices)
        
        return torch.utils.data.Subset(self, combined_indices)

    def __len__(self):
        return len(self.data) + len(self.zoom_womb_data)

    def __getitem__(self, idx):       
        # Determine which dataset to pull from
        if idx < len(self.data):
            data = self.data[idx]
        else:
            data = self.zoom_womb_data[idx - len(self.data)]
        volume, _   = read_nifti(data['filename'])
        sid         = data['sid']
        zf          = 1
        zoom_womb   = data['zm']
        
        ## Put subject into TorchIO format
        subject = tio.Subject(raw_volume=tio.ScalarImage(tensor=volume[None, ...]))

        ## Determine normalization type
        if self.opts.norm_type == 'percentile':
            necessary = NormalizeByPercentile()
        elif self.opts.norm_type == 'percentile_window':
            necessary = tio.RescaleIntensity((0, 1), (1., 99.0))
        
        ## Determine the augmentations
        if self.stage == 'train' and self.opts.custom_augmentation is True:# and zoom_womb is False:
            if random() < self.opts.augmentation_prob:
                p = 0.8
                # if random() < 0.5: then apply augmentations, no anistropy and then zoom
                if random() < 0.5:

                    if self.opts.zoom > 0:
                        
                        if random() < self.opts.zoom and zoom_womb is False:
                            # Sample a random zoom factor
                            zf              = uniform(0.55, 0.65)
                            zoomer          = Zoom(zf=zf)
                            
                            # Apply normalization and zoom
                            subject         = necessary(subject)
                            subject         = zoomer(subject)

                            # Define some augmentations
                            augmentations   = tio.Compose([
                                                tio.RandomNoise(std=(0.02, 0.03), p=p),
                                                tio.Clamp(out_min=0),
                                                ])
                        
                            # Apply the augmentations
                            transformed     = augmentations(subject)
                        
                            # Convert the subject data to numpy
                            volume          = transformed['raw_volume'].data[0].numpy()
                            joint_coord     = data['label'] * zf
                        else:
                            # Def the augmentations // no zoom
                            augmentations   = tio.Compose([
                                necessary,
                                tio.RandomNoise(std=(0.02, 0.03), p=p),
                                tio.RandomSpike(num_spikes=(4), intensity=0.3, p=p),
                                tio.OneOf({tio.RandomBiasField(coefficients=(0.1, 0.165), p=p),tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)}),
                                tio.Clamp(out_min=0),
                            ])
                            
                            # Apply the augmentations
                            transformed         = augmentations(subject)
                        
                            # Convert the subject data to numpy
                            volume              = transformed['raw_volume'].data[0].numpy()
                            zf                  = 1
                            joint_coord         = data['label']
                    
                else:
                    random_anisotropy = tio.OneOf({tio.RandomAnisotropy(axes=(0), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(1), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(2), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(0, 1), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(0, 2), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(1, 2), downsampling=(1.5,2), p=p),
                            tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5,2), p=p),
                        })
                    
                    ## Construct the transformations
                    augmentations = tio.Compose([
                        necessary,
                        tio.RandomNoise(std=(0.02, 0.03), p=p),
                        tio.OneOf({tio.RandomBiasField(coefficients=(0.1, 0.165), p=p),tio.RandomGamma(log_gamma=(-0.25, 0.25), p=p)}),
                        random_anisotropy,
                        tio.Clamp(out_min=0),
                    ])
                    
                    ## Choose whether to augment or not
                    transforms = augmentations
                    
                    ## Apply the transformations
                    transformed = transforms(subject)
                    
                    # Convert the subject data to numpy
                    volume      = transformed['raw_volume'].data[0].numpy()
                    joint_coord = data['label']
                    zf          = 1
            # No augmentation
            else:
                # Apply normalization
                transformed = necessary(subject)
                volume      = transformed['raw_volume'].data[0].numpy()
                joint_coord = data['label']
            
        else:
            # If we are using the zoom_womb dataset
            if zoom_womb is True and self.stage == 'train':
                
                transformed = necessary(subject)
                volume      = transformed['raw_volume'].data[0].numpy()
                joint_coord = (data['label'] * zf)
                zf          = 1
            
            # If we are using the baseline model
            elif self.opts.baseline is True and self.stage == 'train' and random() < self.opts.zoom:
                if self.opts.zoom <= 1:
                    zf = uniform(1/self.opts.zoom_factor, self.opts.zoom_factor)
                else:
                    zf = self.opts.zoom_factor
                zoomer          = Zoom(zf=zf)
                transformed     = zoomer(subject)
                transformed     = necessary(subject)
                volume          = transformed['raw_volume'].data[0].numpy()
                joint_coord     = data['label'] * zf
                zf              = 1 # because Junshen doesn't scale sigma
            
            # If we are doing nothing to the data // validation or test
            else:
                transformed = necessary(subject)
                volume      = transformed['raw_volume'].data[0].numpy()
                joint_coord = data['label']
                zf          = 1

        if self.opts.loss == 'adaptive':
            zf = 1

        # Crop the image
        volume, origin = self.crop(volume) 
        
        # If temporal training, get the previous volume            
        volume_prev = np.zeros_like(volume)
        
        # Get the heatmaps if the stage is train or val
        if self.stage == 'train' or self.stage == 'val':
            heatmap = self.gen_hmap(joint_coord, origin, zf)
        
        volume      = np.expand_dims(volume, axis=0)
        if volume_prev is not None:
            volume_prev = np.zeros_like(volume)           
        
        # Apply rotations if the stage is train
        if self.stage == 'train':
            if self.opts.rot is True:
                volume, volume_prev, heatmap = random_rot(volume, volume_prev, heatmap)
                volume          = volume.copy()
                heatmap         = heatmap.copy()
                volume_prev     = volume_prev.copy()
                
            if self.opts.junshen_scale is not None:
                volume = volume ** uniform(
                1 - self.opts.junshen_scale, 1 + self.opts.junshen_scale
            )
        
        if self.stage == 'test':
            return volume, joint_coord, data['sid']
        else:
            return volume, (0 if volume_prev is None else volume_prev), heatmap
        
        
    # cropping function
    def crop(self, volume, origin=None):
        cs = self.opts.crop_size
        if self.stage == 'test':
            if self.opts.unet_type == 'small':
                depth = self.opts.depth + 1
            else:
                depth = self.opts.depth
            
            f = 2**depth
            pad_width = [(0, ceil(s/f)*f-s) for s in volume.shape]
            volume = np.pad(volume, pad_width, mode='constant')
            x_0 = y_0 = z_0 = 0
        else:
            if any(s < cs for s in volume.shape):
                pad_width = [(0, max(0, cs-s)) for s in volume.shape]
                volume = np.pad(volume, pad_width, mode='constant')
            if origin:
                x_0, y_0, z_0 = origin
            else:
                x_0, y_0, z_0 = randint(0, volume.shape[1] - cs), randint(0, volume.shape[0] - cs), randint(0, volume.shape[2] - cs) # close interval
            volume = volume[y_0:y_0+cs, x_0:x_0+cs, z_0:z_0+cs]
        return volume, (x_0, y_0, z_0)
    
    # heatmap generating function
    def gen_hmap(self, joint, origin, zf=1): # Added zf as an argument for zoom factor, may need to change this

        joint = np.around(joint).astype(np.int32)

        cs = self.opts.crop_size
        x_0, y_0, z_0 = origin
        y_range = np.reshape(np.arange(y_0+1, y_0+cs+1, dtype=np.float32), (1,-1,1,1))
        x_range = np.reshape(np.arange(x_0+1, x_0+cs+1, dtype=np.float32), (1,1,-1,1))
        z_range = np.reshape(np.arange(z_0+1, z_0+cs+1, dtype=np.float32), (1,1,1,-1))

        x_label, y_label, z_label = np.reshape(joint, (3,-1,1,1,1))
        dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
        dd = dx**2 + dy**2 + dz**2

        heatmap = self.opts.mag * (2.0 ** 3) / (2 ** 3) * np.exp(-0.5 / (self.opts.sigma * zf) ** 2 * dd)
        return heatmap.astype(np.float32, copy=False)
      

if __name__ == '__main__':
    # Test the data
    opts = {'rawdata_path': '/data/vision/polina/projects/fetal/common-data/pose/epis',
            'label_path': '/data/vision/polina/projects/fetal/common-data/pose/SeboPoseLabel',
            'custom_augmentation': True,
            'baseline': False,
            'use_zoom_womb': False,
            'crop_size': 96,
            'mag': 10.0,
            'sigma': 2.0,
            'rot': False,
            'zoom_factor': 1.5,
            'augmentation_prob': 0.90,
            'nJoints': 15,
            'junshen_scale': None,
            'norm_type': 'percentile',
            'dataset_size': 8000,
            'zoom': 0.5,
            'noise': True,
            'spike': True,
            'bfield': True,
            'gamma': True,
            'anisotropy': True,
            }
    opts                = type('opts', (object,), opts)

    train_dataset       = Dataset('train', opts)
    train_dataloader    = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)#, collate_fn=collate_fn)
    
    for batch in train_dataloader:
        volume, labels = batch

        print(volume.shape, labels.shape)
        
        # print max and min values
        print(f'max: {volume.max()}, min: {volume.min()}, | max of heatmap: {labels.max()}, min of heatmap: {labels.min()}')
        print(f'volume shape: {volume.shape}, label shape: {labels.shape}')

        # Save one instance of the augmentations
        nib.save(nib.Nifti1Image(volume[0, 0].numpy(), np.eye(4)), 'outs/img.nii.gz')
        label = np.zeros(volume[0,0].shape)
        for i in range(15):
            label += labels[0, i].numpy()
        nib.save(nib.Nifti1Image(label, np.eye(4)), 'outs/lab.nii.gz')
        break
    