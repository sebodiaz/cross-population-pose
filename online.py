import torch
import torchio as tio
from torch.utils.data import DataLoader
import numpy as np
import scipy.io
import scipy
import os
from tqdm import tqdm
import nibabel as nib
import time
from monai.networks.layers import separable_filtering # added for faster convultion
import torch.nn.functional as F
import yaml
import random

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

zoom_womb = ['ZM1', 'ZM2', 'ZM3', 'ZM4', 'ZM5', 'ZM6', 'ZM7', 'ZM8', 'ZM9', 'ZM10']

data_val = [
    '092117L', '092117S', '052516', '032818', '041818', '022618',
    '080217', '061217', '110214', '040716', '083115', '071218', '040417'
]
data_test = [
    '100317L', '100317S', '120717', '090517L', '050318L', '051817',
    '062117', '102617', '092817L', '040218', '041017', '101317',
    '082517L', '041318L',
]

# Configuration
SAMPLES_PER_VOLUME  = 3        # Patches to extract per volume
MAX_QUEUE_LENGTH    = 100       # Max batches stored in the queue

label_dictionary = {
        'l_ankle': '',
        'r_ankle': '',
        'l_knee': '',
        'r_knee': '',
        'bladder': '',
        'l_elbow': '',
        'r_elbow': '',
        'l_eye': '',
        'r_eye': '',
        'l_hip': '',
        'r_hip': '',
        'l_shoulder': '',
        'r_shoulder': '',
        'l_wrist': '',
        'r_wrist': ''
    }

# Gaussian kernel
def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    half_size = (kernel_size - 1) / 2
    sigma = torch.tensor(kernel_size / 3.0)
    x = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize to sum to 1
    return kernel

# Triangle kernel
def make_triangle_kernel(kernel_size: int) -> torch.Tensor:
    '''
    This is a triangle kernel, not a gaussian kernel. Just here for illustration
    purposes.
    '''
    fsize = (kernel_size + 1) // 2
    if fsize % 2 == 0:
        fsize -= 1
    f = torch.ones((1, 1, fsize), dtype=torch.float).div(fsize)
    padding = (kernel_size - fsize) // 2 + fsize // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)

# Gaussian class for precomputing // really slow... I do not usually use this, but *technically* it is more stable
class Gauss(torch.nn.Module):
    """
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        kernel_type: str = "triangle",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
        """
        super().__init__()

        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        if kernel_type == "gaussian":
            self.kernel = make_gaussian_kernel(self.kernel_size)
        elif kernel_type == "triangle":
            self.kernel = make_triangle_kernel(self.kernel_size)

        self.kernel.require_grads = False
        self.kernel_vol = self.get_kernel_vol()

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return torch.sum(vol)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}")

        kernel, kernel_vol = self.kernel.to(pred), self.kernel_vol.to(pred)
        kernels = [kernel] * self.ndim
        # sum over kernel
        p_sum = separable_filtering(pred, kernels=kernels)
        #'0', p_sum.max(), p_sum.min())
        # average over kernel
        p_avg = p_sum / kernel_vol
        
        # the max in each channel should be 10
        #print('1', p_avg.max(), p_avg.min())
        p_avg = p_avg / p_avg.max()
        #print('2', p_avg.max(), p_avg.min())
        
        
        return p_avg * 10

gauss = Gauss(spatial_dims=3, kernel_size=7, kernel_type="gaussian")

class FetalPoseDenseGPT(tio.SubjectsDataset):
    def __init__(self, data_partition_file, opts, stage="train", transform=None, proc_type='neel'):
        # Store configuration first
        self.opts = opts
        self.stage = stage
        self.proc_type = proc_type
        self.transform = transform  # Store transform before super().__init__
        
        # Load data partition from YAML
        with open(data_partition_file, 'r') as f:
            self.partition_data = yaml.safe_load(f)
        
        # Create subjects list with lazy loading
        subjects = self._create_subjects_lazy()
        
        # Initialize parent class
        super().__init__(subjects)
    
    def _create_subjects_lazy(self):
        """Create subjects with lazy loading"""
        stage_dict = self.partition_data[f'data_{self.stage}']
        
        if self.stage == "train" and self.opts.use_zoom_womb:
            print('Using zoom_womb data')
            stage_dict.update(self.partition_data['zoom_womb'])
        
        subjects = []
        
        for folder, sampling_density in tqdm(stage_dict.items(), desc=f"Creating {self.stage} subjects", ncols=70):
            max_samples = int(self.opts.base_samples_per_subject * sampling_density)
            data_dir = os.path.join(self.opts.rawdata_path, folder)
            label_file = os.path.join(self.opts.label_path, f"{folder}.mat")
            
            nii_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
            random.shuffle(nii_files)
            
            max_samples = min(max_samples, len(nii_files))
            
            for nii_file in nii_files[:max_samples]:
                image_path = os.path.join(data_dir, nii_file)
                idx = int(nii_file.split('_')[1].split('.')[0])
                
                # Create a Subject with LazyImage
                if self.proc_type == 'sebo':
                    subject = tio.Subject(
                        image=tio.ScalarImage(image_path),
                        _label_file=label_file,
                        _idx=idx,
                        _proc_type='sebo'
                    )
                elif self.proc_type == 'neel':
                    subject = tio.Subject(
                        image=tio.ScalarImage(image_path),
                        _label_file=label_file,
                        _idx=idx,
                        _proc_type='neel'
                    )
                
                subjects.append(subject)
        
        return subjects
    
    def __getitem__(self, index):
        """Load and process data when accessed"""
        subject = super().__getitem__(index)
        
        # Load labels if not already processed
        if '_label_file' in subject:
            labels = scipy.io.loadmat(subject._label_file)['joint_coord']
            idx = subject._idx
            proc_type = subject._proc_type
            
            # Remove temporary attributes
            del subject._label_file
            del subject._idx
            del subject._proc_type
            
            if proc_type == 'sebo':
                label_vol = self._generate_heatmap(subject.image.shape[1:], 
                                                 labels[idx],
                                                 mag=self.opts.mag, 
                                                 sigma=self.opts.sigma)
                label_vol = torch.from_numpy(label_vol).float().permute(3, 0, 1, 2)
                
                # Add label maps to subject
                affines = subject.image.affine
                for i, key in enumerate(label_dictionary):
                    label = label_vol[i].unsqueeze(0)
                    subject[key] = tio.LabelMap(tensor=label, affine=affines)
                    
            elif proc_type == 'neel':
                subject['label'] = self._create_label_volume(subject.image.path, labels[idx])
        
        # Apply transform if it exists
        if self.transform is not None:
            subject = self.transform(subject)
            
        return subject
    
    @staticmethod
    def _generate_heatmap(volume_shape, joint_coord, mag, sigma):
        crop_size_y, crop_size_x, crop_size_z = volume_shape
        
        # Generate coordinate ranges using torch operations
        y_range = torch.arange(1, crop_size_y + 1, dtype=torch.float32).reshape(-1, 1, 1, 1)
        x_range = torch.arange(1, crop_size_x + 1, dtype=torch.float32).reshape(1, -1, 1, 1)
        z_range = torch.arange(1, crop_size_z + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        
        joint = torch.tensor(joint_coord, dtype=torch.float32).reshape(3, 1, 1, 1, -1)
        x_label, y_label, z_label = joint.unbind(0)
        
        # Calculate distances using torch operations
        dx = x_range - x_label
        dy = y_range - y_label
        dz = z_range - z_label
        dd = dx.pow(2) + dy.pow(2) + dz.pow(2)
        
        if sigma:
            heatmap = mag * ((2.0 / sigma) ** 3) * torch.exp(-0.5 * dd / sigma**2)
        else:
            heatmap = dd
            
        return heatmap.numpy()

class FetalPoseDense(tio.SubjectsDataset):
    def __init__(self, data_partition_file, opts, stage="train", transform=None, proc_type='neel'):
        # Load data partition from YAML
        with open(data_partition_file, 'r') as f:
            self.partition_data = yaml.safe_load(f)
            
        subjects = self._create_subjects(self.partition_data, stage, opts, proc_type)
        super().__init__(subjects, transform=transform)
    
    def _create_subjects(self, partition_data, stage, opts, proc_type):
        # Get the appropriate dataset split
        stage_dict = partition_data[f'data_{stage}']
        
        # Extend with zoom_womb if in training stage and opts.use_zoom_womb is True
        if stage == "train" and opts.use_zoom_womb is True:
            print('Using zoom_womb data')
            stage_dict.update(partition_data['zoom_womb'])
        
        subjects = []
        
        # Loop through folders and create subjects
        for folder, sampling_density in tqdm(stage_dict.items(), desc=f"Creating {stage} data", ncols=70):
            # Calculate number of samples based on sampling density
            max_samples = int(opts.base_samples_per_subject * sampling_density)
            
            # change the directory if zoom_woomb is detected
            if folder in zoom_womb:
                data_dir = os.path.join(opts.zw_path, folder)
            
            # Load data and labels
            data_dir    = os.path.join(opts.rawdata_path, folder)
            label_file  = os.path.join(opts.label_path, f"{folder}.mat")
            labels      = scipy.io.loadmat(label_file)['joint_coord']
            nii_files   = os.listdir(data_dir)

            # Shuffle the nii_files
            random.shuffle(nii_files)  # In-place shuffling
            
            # Check if max_samples exceeds available files
            if max_samples > len(nii_files):
                max_samples = len(nii_files)
            
            # Loop through each file
            for idx_num, nii_file in tqdm(enumerate(nii_files), desc=f"Processing {folder}", total=len(nii_files), disable=True, position=1):
                if not nii_file.endswith('.nii.gz'):
                    continue
                    
                image_path  = os.path.join(data_dir, nii_file)
                image       = tio.ScalarImage(image_path)
                idx         = int(nii_file.split('_')[1].split('.')[0])
                
                if proc_type == 'sebo':
                    label_vol   = self._generate_heatmap(image.shape[1:], labels[idx], mag=opts.mag, sigma=opts.sigma)
                    label_vol   = torch.from_numpy(label_vol).float().permute(3, 0, 1, 2)
                    label_maps  = {}
                    affines     = image.affine
                    
                    for i, key in enumerate(label_dictionary):
                        label   = label_vol[i].unsqueeze(0)
                        label_maps[key] = tio.LabelMap(tensor=label, affine=affines)
                    
                    subject = tio.Subject(
                        image=image,
                        l_ankle=label_maps['l_ankle'],
                        r_ankle=label_maps['r_ankle'],
                        l_knee=label_maps['l_knee'],
                        r_knee=label_maps['r_knee'],
                        bladder=label_maps['bladder'],
                        l_elbow=label_maps['l_elbow'],
                        r_elbow=label_maps['r_elbow'],
                        l_eye=label_maps['l_eye'],
                        r_eye=label_maps['r_eye'],
                        l_hip=label_maps['l_hip'],
                        r_hip=label_maps['r_hip'],
                        l_shoulder=label_maps['l_shoulder'],
                        r_shoulder=label_maps['r_shoulder'],
                        l_wrist=label_maps['l_wrist'],
                        r_wrist=label_maps['r_wrist']
                    )
                elif proc_type == 'neel':
                    subject = tio.Subject(
                        image=image,
                        label=self._create_label_volume(image_path, labels[idx])
                    )
                
                subjects.append(subject)
                
                if idx_num >= max_samples:
                    break
                    
        return subjects
    
    def _generate_heatmap(self, volume_shape, joint_coord, mag, sigma):
        crop_size_y, crop_size_x, crop_size_z = volume_shape
        
        # Generate coordinate ranges
        y_range = np.reshape(np.arange(1, crop_size_y + 1, dtype=np.float32), (-1, 1, 1, 1))
        x_range = np.reshape(np.arange(1, crop_size_x + 1, dtype=np.float32), (1, -1, 1, 1))
        z_range = np.reshape(np.arange(1, crop_size_z + 1, dtype=np.float32), (1, 1, -1, 1))
        
        def gen_hmap(joint):
            x_label, y_label, z_label = np.reshape(joint, (3, 1, 1, 1, -1))
            dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
            dd = dx**2 + dy**2 + dz**2
            if sigma:
                return (
                    mag
                    * ((2.0 / sigma) ** 3)
                    * np.exp((-1.0 / 2.0 / sigma**2) * dd, dtype=np.float32)
                )
            else:
                return dd.astype(np.float32)
        
        generated_heatmap = gen_hmap(joint_coord)

        return generated_heatmap

class FetalPoseFinal(tio.SubjectsDataset):
    def __init__(self, data_partition_file, opts, stage="train", transform=None, proc_type='neel'):
        # Load data partition from YAML
        with open(data_partition_file, 'r') as f:
            self.partition_data = yaml.safe_load(f)
            
        subjects = self._create_subjects(self.partition_data, stage, opts, proc_type)
        super().__init__(subjects, transform=transform)
    
    def _create_subjects(self, partition_data, stage, opts, proc_type):
        # Get the appropriate dataset split
        stage_dict = partition_data[f'data_{stage}']
        
        # Extend with zoom_womb if in training stage and opts.use_zoom_womb is True
        if stage == "train" and opts.use_zoom_womb is True:
            print('Using zoom_womb data')
            stage_dict.update(partition_data['zoom_womb'])
        
        subjects = []
        # Loop through folders and create subjects
        for folder, sampling_density in tqdm(stage_dict.items(), desc=f"Creating {stage} data", ncols=70):
            # Check if folder is in zoom_womb
            if folder in zoom_womb:
                data_dir    = os.path.join(opts.zw_path, folder)
                label_dir   = os.path.join(opts.zwlabel_path, folder)
            else:
                data_dir    = os.path.join(opts.rawdata_path, folder)
                label_dir   = os.path.join(opts.vollabel_path, folder)
            
            
            
            # Calculate number of samples based on sampling density
            max_samples = int(opts.base_samples_per_subject * sampling_density)
            
            # Load data and labels
            
            
            nii_files   = os.listdir(data_dir)

            # Shuffle the nii_files
            random.shuffle(nii_files)  # In-place shuffling
            
            # Check if max_samples exceeds available files
            if max_samples > len(nii_files):
                max_samples = len(nii_files)
            
            # Loop through each file
            for idx_num, nii_file in tqdm(enumerate(nii_files), desc=f"Processing {folder}", total=len(nii_files), disable=True, position=1):
                if not nii_file.endswith('.nii.gz'):
                    continue
                    
                image_path  = os.path.join(data_dir, nii_file)
                label_path  = os.path.join(label_dir, nii_file)
                
                if proc_type == 'sebo':
                                    
                    subject = tio.Subject(
                        image=tio.ScalarImage(image_path),
                        label=tio.LabelMap(label_path)
                    )
                
                subjects.append(subject)
                
                if idx_num >= max_samples:
                    break
                    
        return subjects

# Pose Class -- most optimal
class FetalPoseDataset(tio.SubjectsDataset):
    def __init__(self, folders, opts, stage="train", transform=None, proc_type='neel'):
        subjects = self._create_subjects(folders, stage, opts, proc_type)
        super().__init__(subjects, transform=transform)
        
    def _create_subjects(self, folders, stage, opts, proc_type):
        # Create a list of subjects
        stage_list = data_train if stage == "train" else data_val
        
        # Extend with zoom_womb if in training stage and opts.use_zoom_womb is True
        if stage == "train" and opts.use_zoom_womb is True:
            print('Using zoom_womb data')
            stage_list.extend(zoom_womb)
        subjects = []
        
        # Loop through folders and create subjects
        for folder in tqdm(folders, desc=f"Creating {stage} data", ncols=70, disable=False, position=0):
            if folder not in stage_list:
                continue
            
            # Load data and labels (modify paths as needed)
            data_dir        = os.path.join(opts.rawdata_path, folder)
            label_file      = os.path.join(opts.label_path, f"{folder}.mat")
            labels          = scipy.io.loadmat(label_file)['joint_coord'] # commented out SDD
            nii_files       = (os.listdir(data_dir))

            # Loop through each files
            for idx_num, nii_file in tqdm(enumerate(nii_files), desc=f"Processing {folder}", total=len(nii_files), disable=True, position=1):
                if not nii_file.endswith('.nii.gz'):
                    continue
                image_path  = os.path.join(data_dir, nii_file)
                image       = tio.ScalarImage(image_path) #; print(image.affine)
                
                idx = int(nii_file.split('_')[1].split('.')[0])

                # loop through each keypoint
                if proc_type == 'sebo':
                    label_vol      = self._generate_heatmap(image.shape[1:], labels[idx], mag=opts.mag, sigma=opts.sigma)
                    label_vol      = torch.from_numpy(label_vol).float().permute(3, 0, 1, 2) #.unsqueeze(0)
                    label_maps = {}
                    affines = image.affine
                    for i, key in enumerate(label_dictionary):
                        label = label_vol[i].unsqueeze(0)
                        label_maps[key] = tio.LabelMap(tensor=label, affine=affines)

                    # add to the subject
                    subject = tio.Subject(
                        image  = image,
                        l_ankle=label_maps['l_ankle'],
                        r_ankle=label_maps['r_ankle'],
                        l_knee=label_maps['l_knee'],
                        r_knee=label_maps['r_knee'],
                        bladder=label_maps['bladder'],
                        l_elbow=label_maps['l_elbow'],
                        r_elbow=label_maps['r_elbow'],
                        l_eye=label_maps['l_eye'],
                        r_eye=label_maps['r_eye'],
                        l_hip=label_maps['l_hip'],
                        r_hip=label_maps['r_hip'],
                        l_shoulder=label_maps['l_shoulder'],
                        r_shoulder=label_maps['r_shoulder'],
                        l_wrist=label_maps['l_wrist'],
                        r_wrist=label_maps['r_wrist']
                    )
                elif proc_type == 'neel':
                    subject = tio.Subject(
                        image=image,
                        label=self._create_label_volume(image_path, labels[idx]))      
            
                subjects.append(subject)
                
                if idx_num > opts.base_samples_per_subject:
                    break

        return subjects

    def _create_label_volume(self, image_path, label):
        image       = tio.ScalarImage(image_path)
        label_vol   = np.zeros(image.shape[1:], dtype=np.int64)  # Remove channel dim
        for j in range(label.shape[1]):
            coords  = label[:, j].astype(int)
            x_slice = slice(max(0, coords[0]-1), min(label_vol.shape[0], coords[0]+2))
            y_slice = slice(max(0, coords[1]-1), min(label_vol.shape[1], coords[1]+2))
            z_slice = slice(max(0, coords[2]-1), min(label_vol.shape[2], coords[2]+2))
            label_vol[y_slice, x_slice, z_slice] = j + 1
        return tio.LabelMap(tensor=label_vol[np.newaxis], affine=image.affine)  # Add channel dim
    
      
    def _generate_heatmap(self, volume_shape, joint_coord, mag, sigma):
        crop_size_y, crop_size_x, crop_size_z = volume_shape
        
        # Generate coordinate ranges
        y_range = np.reshape(np.arange(1, crop_size_y + 1, dtype=np.float32), (-1, 1, 1, 1))
        x_range = np.reshape(np.arange(1, crop_size_x + 1, dtype=np.float32), (1, -1, 1, 1))
        z_range = np.reshape(np.arange(1, crop_size_z + 1, dtype=np.float32), (1, 1, -1, 1))
        
        def gen_hmap(joint):
            x_label, y_label, z_label = np.reshape(joint, (3, 1, 1, 1, -1))
            dx, dy, dz = x_range - x_label, y_range - y_label, z_range - z_label
            dd = dx**2 + dy**2 + dz**2
            if sigma:
                return (
                    mag
                    * ((2.0 / sigma) ** 3)
                    * np.exp((-1.0 / 2.0 / sigma**2) * dd, dtype=np.float32)
                )
            else:
                return dd.astype(np.float32)
        
        generated_heatmap = gen_hmap(joint_coord)

        return generated_heatmap
          

def save_vol(vol, title='volume', batches=1):
    
    for i in range(batches):
        vol1 = vol[i, 0].numpy().astype(np.float32)
        nii = nib.Nifti1Image(vol1, np.eye(4))
        nib.save(nii, f'outs/{title}_{i}.nii.gz')
    return

def save_label(label, title='label', batches=1):
    for i in range(batches):
        label_vol = np.zeros(label.shape[2:], dtype=np.float32)
        for j in range(15):
            label_vol += label[i, j].numpy()
        nii = nib.Nifti1Image(label_vol, np.eye(4))
        nib.save(nii, f'outs/{title}_{i}.nii.gz')
    return

def center_of_mass_v2(labels):
    for j in range(labels.shape[0]):
        for i in range(labels.shape[1]):
            coords = torch.where(labels[j, i] == 1)
            
            # Convert coords to float for mean calculation
            x = coords[0].float().mean()
            y = coords[1].float().mean()
            z = coords[2].float().mean()
            
            # Set the corresponding label to 0
            labels[j, i] = 0
            
            # If any coordinate is NaN, skip to the next iteration
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            
            # Convert x, y, z to integers (after calculating the mean) and set the new label
            labels[j, i, int(x), int(y), int(z)] = 1
            
    return labels

def center_of_mass(labels):
    # Find indices where labels == 1
    indices = torch.nonzero(labels, as_tuple=False)  # Shape: (N, 5) where (batch, channel, x, y, z)
    
    if indices.shape[0] == 0:
        return labels  # No foreground pixels, return as is

    batch_idx, channel_idx, x, y, z = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], indices[:, 4]

    # Compute center of mass for each (batch, channel) group
    unique_batches_channels = torch.unique(indices[:, :2], dim=0)  # Unique (batch, channel) pairs
    new_labels = torch.zeros_like(labels)  # Initialize output tensor

    for b, c in unique_batches_channels:
        mask = (batch_idx == b) & (channel_idx == c)
        x_mean = x[mask].float().mean().round().long()
        y_mean = y[mask].float().mean().round().long()
        z_mean = z[mask].float().mean().round().long()

        # Check bounds to avoid index errors
        if 0 <= x_mean < labels.shape[2] and 0 <= y_mean < labels.shape[3] and 0 <= z_mean < labels.shape[4]:
            new_labels[b, c, x_mean, y_mean, z_mean] = 1  # Set center of mass

    return new_labels

def produce_labels(labels, num_classes=16, sigma=2, kernel_size=5):
    with torch.no_grad():
        # One-hot encoding
        labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)[:, 0, ...]
        labels = labels.permute(0, 4, 1, 2, 3).float()[:, 1:]  

        # Apply 1D Gaussian smoothing 
        labels = gauss(labels)
    
    return labels

def get_online_dloader_og(opts, stage="train"):
    folders = globals()[f"data_{stage}"]
    dataset = FetalPoseDataset(folders, opts=opts, stage=stage, transform=get_augmentations(opts), proc_type=opts.proc_type)
    sampler = tio.data.UniformSampler(opts.online_crop_size)
    queue   = tio.Queue(
                dataset,
                max_length=opts.max_queue_length,
                samples_per_volume=opts.samples_per_volume,
                sampler=sampler,
                num_workers=opts.num_workers,
            )
    dataloader = DataLoader(
                queue,
                batch_size=opts.online_batch_size,
                num_workers=0,  # Already handled by the queue
                pin_memory=True,
            )
    return dataloader

def get_online_dloader(opts, stage="train"):
    dataset = FetalPoseFinal(
        opts.data_partition_file,
        opts=opts,
        stage=stage,
        transform=get_augmentations(opts),
        proc_type=opts.proc_type
    )
    
    sampler = tio.data.UniformSampler(opts.online_crop_size)
    queue = tio.Queue(
        dataset,
        max_length=opts.max_queue_length,
        samples_per_volume=opts.samples_per_volume,
        sampler=sampler,
        num_workers=opts.num_workers,
    )
    
    dataloader = DataLoader(
        queue,
        batch_size=opts.online_batch_size,
        num_workers=0,  # Already handled by the queue
        pin_memory=True,
    )
    
    return dataloader

# Define a custom normalization class based on TorchIO's Transform class
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
        data = transformed['image'].data
        
        # Calculate percentile of non-zero values for the first channel
        percentile_fac = np.percentile(data[0][data[0] > 0], self.percentile)
        
        if self.scale is True: # scale from 0 to 1
            # Normalize the data
            subject['image'].set_data(data / percentile_fac)
            
            # Scale the data
            subject['image'].set_data((data - data.min()) / (data.max() - data.min()))
        else:
            # Normalize the data
            subject['image'].set_data(data / percentile_fac)
        
        return subject
    
def get_augmentations(opts):
    # Probability of applying any given augmentation
    p = opts.augmentation_prob
    
    # Necessary augmentations
    init_norm   = tio.RescaleIntensity((0, 1), (1, 99.0))
    #init_norm   = NormalizeByPercentile(percentile=99, scale=False)
    croporpad   = tio.CropOrPad((opts.online_crop_size))
    
    # Non-linear # different interpolations are necessary
    if opts.proc_type == 'sebo':
        affine      = tio.RandomAffine(scales=(0.55, 0.65), degrees=(-180, 180), translation=10, p=p, label_interpolation='bspline') # 'bspline'
    elif opts.proc_type == 'neel':
        affine      = tio.RandomAffine(scales=(0.55, 0.65), degrees=(-180, 180), translation=10, p=p)
    nonlinear   = tio.Compose([affine])
    
    # Additive
    noise       = tio.RandomNoise(std=(0.015, 0.025), p=p)
    gamma       = tio.RandomGamma(log_gamma=(-0.35, 0.35), p=p)
    
    biasfield   = tio.RandomBiasField(coefficients=0.65, p=p)
    
    intensity   = tio.OneOf([gamma, biasfield])
    
    # Anisotropic
    resolution  = tio.OneOf([
        tio.RandomAnisotropy(axes=(0, 1), downsampling=(2, 3), p=p),
        tio.RandomAnisotropy(axes=(0, 2), downsampling=(2, 3), p=p),
        tio.RandomAnisotropy(axes=(1, 2), downsampling=(2, 3), p=p),
    ])
    
    # Resample
    resample1   = tio.Resample((6, 6, 6))
    resample2   = tio.Resample((3, 3, 3))
    resampler   = tio.Compose([resample1, resample2], p=p)
    resolutions = tio.OneOf({
        resolution: 0.85,
        resampler: 0.15})
    
    # Clamp
    clamp       = tio.Clamp(out_min=0)

    # Compose augmentations
    aggressive  = tio.Compose([init_norm, intensity, noise, nonlinear, resolutions, noise, clamp, croporpad])
    conservative= tio.Compose([init_norm, croporpad])
    
    # Set the augmentation mixing percentages
    augmentations=tio.OneOf({
        conservative: 0.15,
        aggressive: 0.85
    })

    return augmentations

def stack_labels(batch):
    # Use torch.stack to efficiently concatenate along the second dimension
    return torch.stack([batch[f'{label}'][tio.DATA][:, 0] for label in label_dictionary], dim=1)
    

# Helper script to visualize augmentations adn test the speed    
if __name__ == '__main__':
    import multiprocessing
    from time import time
    import matplotlib.pyplot as plt

    # Configuration object
    opts = type('opts', (object,), {
        'online_batch_size': 4,
        'online_crop_size': 64,
        'use_zoom_womb': False,
        'rawdata_path': '/unborn/shared/fetal_pose/fetalEPI',#'/data/vision/polina/projects/fetal/common-data/pose/epis',
        'label_path': '/unborn/shared/SeboPoseLabel',#'/data/vision/polina/projects/fetal/common-data/pose/SeboPoseLabel',
        'augmentation_prob': 0.5,
        'sigma': 2,
        'mag': 10,
        'data_partition_file': 'data_partition.yml',
        'max_queue_length': 5,
        'samples_per_volume': 3,
        'num_workers': 4,
        'proc_type': 'sebo',
        'base_samples_per_subject': 10,
    })
    
    # Create dataloader
    dataloader = get_online_dloader(opts, stage="train")
    print('Dataloader created')
    
    # Performance tracking
    total_time = 0
    for i, batch in enumerate(dataloader):
        if i == 0:
            start = time()
        
        inputs = batch['image'][tio.DATA]
        if opts.proc_type == 'neel':
            labs = batch['label'][tio.DATA]

            labels = produce_labels(labs.long())
        else:
            labels = batch['label'][tio.DATA]  
            #labels = stack_labels(batch)

        print(f'Batch processed - Time: {time()-start}')
        
        # Save volumes for inspection
        if i == 0 or i == 1:
            save_vol(inputs, title='input', batches=i)
            save_label(labels, title='label', batches=i)
            save_vol(inputs, title='input', batches=i+1)
            save_label(labels, title='label', batches=i+1)
        total_time += 1
    
        # Break after first batch for testing
    tot = time()-start
    print(f'Total time: {tot / total_time}')
