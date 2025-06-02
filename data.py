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
    
    def __init__(self, zf, order=1):
        super().__init__()
        self.zf     = zf
        self.order  = order
    def apply_transform(self, subject):
        # Create a copy of the subject to avoid modifying the original
        transformed = subject.copy()

        # Loop over all the volumes in the subject
        for image in subject.keys():
            subject[f'{image}'].set_data(zoom(transformed[f'{image}'].data, self.zf, order=self.order))

        return subject

# Define the ablation class // probably the one I will end up with
class Dataset(torch.utils.data.Dataset):
    def __init__(self, stage, opts):
        #############
        ## STAGING ##
        #############

        self.stage      = stage
        self.opts       = opts
        self.subjects   = sorted(globals()['data_' + stage])
        self.norm       = NormalizeByPercentile()
        
        # hardcode the segmentation file for now
        self.segmentation_file = "/data/vision/polina/projects/fetal/common-data/pose/body_segs_dates_cropped/"

        ## Determine whether to use `zoom womb` or `fetal inpainting`... they are the same thing.
        if self.opts.use_fetal_inpainting is True and self.stage == 'train':
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
                base_fname = os.path.basename(filename)
                seg_path   = os.path.join(self.segmentation_file, subject, base_fname)
                self.data.append({'foldername': subject, 'filename': filename, 'label': label, 'sid': dn, 'zm': zw, 'segmentation': seg_path})
        
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
        zoom_factor = 1
        shared_transforms = []
        per_image_transforms = []

        if stage == 'train' and self.opts.custom_augmentation and not zoom_womb:
            if random() < self.opts.augmentation_prob:
                if random() < 0.5:
                    if self.opts.zoom > 0 and random() < self.opts.zoom:
                        zoom_factor = uniform(0.55, 0.65)
                        # Create two Zoom transforms: one for volume, one for seg
                        zoomer_vol = Zoom(zf=zoom_factor, order=1)
                        zoomer_seg = Zoom(zf=zoom_factor, order=0)

                        # Apply zoom immediately to subject parts
                        subject['raw_volume'] = zoomer_vol(subject['raw_volume'])
                        subject['segmentation'] = zoomer_seg(subject['segmentation'])

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
                    shared_transforms.append(self.augmentations['anisotropy'])
                    per_image_transforms.append(self.augmentations['noise'])
                    per_image_transforms.append(self.augmentations['clamp'])
        else:
            shared_transforms.append(self.augmentations['clamp'])

        # Compose and apply shared transforms (not zoom, already applied)
        shared_pipeline = tio.Compose(shared_transforms)
        subject = shared_pipeline(subject)

        # Per-image transforms for each image (noise, clamp, etc)
        per_image_pipeline = tio.Compose(per_image_transforms)
        for image_name in subject.get_images_dict():
            transformed_image = per_image_pipeline(subject[image_name])
            subject[image_name].set_data(transformed_image.data)

        # Update joint coordinates using zoom factor
        joint_coord = label * zoom_factor
        return subject, joint_coord, zoom_factor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data            = self.data[idx]
        volume, _       = read_nifti(data['filename'])
        sid             = data['sid']
        zoom_womb       = data['zm']
        segmentation, _ = read_nifti(data['segmentation'])

        ## Load the subject using TorchIO ##
        #subject_dict                = {}
        #subject_dict["raw_volume"]  = tio.ScalarImage(tensor=torch.tensor(volume[None, ...]))
        subject_dict = {
        "raw_volume": tio.ScalarImage(tensor=torch.tensor(volume[None, ...])),
        "segmentation": tio.LabelMap(tensor=torch.tensor(segmentation[None, ...]))
        }
        subject                     = tio.Subject(**subject_dict)

        # Normalize the subject
        subject                     = self.norm(subject)
        
        # Apply the shared and per-image augmentation pipeline only once.
        subject, joint_coord, zoom_factor = self.apply_augmentations(subject, data['label'], zoom_womb, self.stage)
        
        # if opts.baseline is True
        if self.opts.baseline and self.stage == 'train':
            if random() < self.opts.zoom:
                zoom_factor = uniform(1 / self.opts.zoom_factor, self.opts.zoom_factor)
            else:
                zoom_factor = self.opts.zoom_factor
            aug_volume = Zoom(zf=zoom_factor)(subject)
            
        
        # Extract augmented volumes
        aug_volume          = subject['raw_volume'].data[0].numpy()
        aug_segmentation    = subject['segmentation'].data[0].numpy()

        aug_volume, origin  = self.crop(aug_volume)
        aug_segmentation, _ = self.crop(aug_segmentation, origin)
        

        # Generate heatmap if in train and val
        if self.stage == 'train':
            heatmap = self.gen_hmap(joint_coord, origin, zoom_factor)
                
        # Ensure primary volume has a channel dimension
        aug_volume       = np.expand_dims(aug_volume, axis=0)
        aug_segmentation = np.expand_dims(aug_segmentation, axis=0)

        if self.stage == 'train':
            
            # Rotate
            if self.opts.rot and random() < self.opts.augmentation_prob and self.stage == 'train':
                aug_volume, heatmap, aug_segmentation = random_rot(aug_volume, heatmap, aug_segmentation)
                aug_volume                            = aug_volume.copy()
                heatmap                               = heatmap.copy()
                aug_segmentation                      = aug_segmentation.copy()
                
            # Baseline's gamma augmentations
            if self.opts.baseline_gamma is not None:
                scale_factor = uniform(1 - self.opts.baseline_gamma, 1 + self.opts.baseline_gamma)
                aug_volume   = aug_volume ** scale_factor
        
        ## Returning for single frame training ##
        if self.stage == 'train':
            return aug_volume, heatmap, aug_segmentation
        if self.stage == 'val':
            return aug_volume, joint_coord, aug_segmentation
        if self.stage == 'test':
            return aug_volume, joint_coord, data['sid']

    def crop(self, volume, origin=None):
        cs = self.opts.crop_size
        if self.stage in ['test', 'val']:
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


if __name__ == '__main__':
    # Test the data
    opts = {'rawdata_path': '/data/vision/polina/projects/fetal/common-data/pose/epis',
            'label_path': '/data/vision/polina/projects/fetal/common-data/pose/SeboPoseLabel',
            'custom_augmentation': True,
            'baseline': False,
            'use_fetal_inpainting': False,
            'train_seg': True,
            'crop_size': 64,
            'mag': 10.0,
            'sigma': 2.0,
            'rot': True,
            'zoom_factor': 1.5,
            'augmentation_prob': 1.,
            'baseline_gamma': None,
            'nJoints': 15,
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
        volume, labels, segmentations = batch

        print(volume.shape, labels.shape, segmentations.shape)
        
        # print max and min values
        print(f'max: {volume.max()}, min: {volume.min()}, | max of heatmap: {labels.max()}, min of heatmap: {labels.min()}')
        print(f'volume shape: {volume.shape}, label shape: {labels.shape}')

        # Save one instance of the augmentations
        nib.save(nib.Nifti1Image(volume[0, 0].numpy(), np.eye(4)), 'outs/img.nii.gz')
        label = np.zeros(volume[0,0].shape)
        for i in range(15):
            label += labels[0, i].numpy()
        nib.save(nib.Nifti1Image(label, np.eye(4)), 'outs/lab.nii.gz')
        nib.save(nib.Nifti1Image(segmentations[0, 0].numpy(), np.eye(4)), 'outs/seg.nii.gz')
        
        break
    