"""
Utility functions.

Originally wrriten by #### 01/07/2025. Some functions are copy/pasted from Molin Zhang and Junshen Xu.

"""


# Import necessary libraries
import torch
import numpy as np
import torch.utils
import wandb
from shutil import copy2
import glob
import yaml
import os
from tqdm import tqdm
from math import ceil
import scipy.io as io
import random
import torchio as tio
import monai
import sklearn

# Transformations
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
        data = transformed.data
        
        # Calculate percentile of non-zero values for the first channel
        percentile_fac = np.percentile(data[0][data[0] > 0], self.percentile)
        
        if self.scale is True: # scale from 0 to 1
            # Normalize the data
            subject.set_data(data / percentile_fac)
            
            # Scale the data
            subject.set_data((data - data.min()) / (data.max() - data.min()))
        else:
            # Normalize the data
            subject.set_data(data / percentile_fac)
        
        return subject


# Define some utility functions
def check_arg(parse_fn=None, valid_list=None, allow_none=False):
    def new_parse_fn(x):
        if allow_none and (x is None or x == 'None'):
            return None
        if parse_fn is not None:
            x = parse_fn(x)
        if valid_list is not None:
            if x not in valid_list:
                raise ValueError(f"Invalid value: {x}. Must be one of {valid_list}.")
        return x
    return new_parse_fn


def copyfiles(src, des, exp):
    mkdir(des)
    for f in glob.glob(os.path.join(src, exp)):
        copy2(f, des)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def save_yaml(path, params_dict, key_to_save=None, key_to_drop=None):
    params_dict = to_dict(params_dict)
    if key_to_save is not None:
        params_dict = dict((k, params_dict[k]) for k in key_to_save)   
    elif key_to_drop is not None:
        params_dict = dict((k, params_dict[k]) for k in params_dict.keys() if k not in key_to_drop)  
    with open(path, 'w', encoding='utf8') as f:
        yaml.dump(params_dict, f, default_flow_style=True, allow_unicode=True)

def to_dict(ns):
    if type(ns) is dict:
        return ns
    else:
        return vars(ns)

def print_args(args):
    args = to_dict(args)
    stage = args.get('stage', 'train')  # Get the current stage (default to 'train')
    
    relevant_args = {}

    # Add general options that apply to all stages
    relevant_args.update({k: v for k, v in args.items() if k in [
        'stage', 'run_name', 'rawdata_path', 'label_path', 
        
    ]})

    if stage == 'train':
        # Add training-specific options
        relevant_args.update({k: v for k, v in args.items() if k in [
            'depth', 'nFeat', 'nJoints', 'optimizer', 'lr', 'weight_decay', 'lr_scheduler', 
            'lr_decay_ep', 'batch_size', 'epochs', 'val_freq', 'joint_consistency', 'jc_weight',
            'gpu_ids', 'ngpu', 'num_workers', 'mag', 'sigma','crop_size', 'custom_augmentation',
            'zoom_factor', 'zoom_prob', 'rot', 'save_path', 
        ]})
        print('------------ Train Options -------------')

    elif stage == 'test':
        # Add test-specific options
        relevant_args.update({k: v for k, v in args.items() if k in [
            'continue_path', 'error_threshold', 'gpu_ids', 'ngpu', 'num_workers'
        ]})
        print('------------ Test Options -------------')
        
    elif stage == 'inference':
        # Add inference-specific options (if any)
        relevant_args.update({k: v for k, v in args.items() if k in [
            'top_k', 'index_type', 'output_vis', 'output_path', 'label_name', 'rawdata_path', 'continue_path'
            # Define inference-specific options here, e.g., 'infer_option1', 'infer_option2'
        ]})
        print('------------ Inference Options -------------')
    
    # Print the relevant options
    for k, v in sorted(relevant_args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

# Get total params
def get_total_params(model):
    return sum(p.numel() for p in model.parameters())

# Fetch the seeds used in that particualr run
def get_seeds():
    return torch.initial_seed(), np.random.get_state()[1][0], random.getstate()[1][0]

# Set the seeds
def set_seeds(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # To ensure the proper seed, print the seed from each of the libraries
    # print(f"Seed: {seed}, torch: {torch.initial_seed()}, numpy: {np.random.get_state()[1][0]}, random: {random.getstate()[1][0]}")
    
# Get the dataloader
def get_dataloaders(opts):
    if opts.stage == 'train':
        train_dl, val_dl = get_offline_dataloader(opts)
        return train_dl, val_dl
    if opts.stage == 'finetune':
        train_dl, _ = get_offline_dataloader(opts)
        return train_dl, _
    elif opts.stage == 'test':
        test_dl = get_offline_dataloader(opts)
        return test_dl
    
def get_offline_dataloader(opts):
    import data
    from torch.utils.data import DataLoader
    
    if opts.stage == 'train':
        
        # Get the validation and training datasets
        train_ds    = data.Dataset('train', opts)
        train_val   = data.Dataset('val', opts)
 
        # if the dataset size is hardcoded, shuffle... it is recommended to use the --seed argument for this
        if opts.dataset_size is not None:
            indices     = torch.randperm(len(train_ds)).tolist()
            rnd_indices = indices[:opts.dataset_size]
            train_ds    = torch.utils.data.Subset(train_ds, rnd_indices)

        # get the dataloaders
        train_dl    = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, pin_memory=True, persistent_workers=True)
        val_dl      = DataLoader(train_val, batch_size=1, shuffle=False, num_workers=opts.num_workers, pin_memory=True, persistent_workers=True, drop_last=False)

        return train_dl, val_dl
    
    elif opts.stage == 'finetune':
        # TODO
        train_dl    = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
        
        return train_dl, None

    elif opts.stage == 'test':
        # create the dataset
        test_ds     = data.Dataset('test', opts)
        test_dl     = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=opts.num_workers)
        
        return test_dl

# Save the model
def save_model_best(model, optimizer, scheduler, epoch, opts):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, f'{opts.save_path}/checkpoints/E{epoch}.pth')

def save_model_latest(model, optimizer, scheduler, epoch, opts):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, f'{opts.save_path}/checkpoints/latest.pth')

# Get amix
def get_amix(opts):
    from model_zoo.small_unet import Unet
    amix = Unet(dimension=3,
                input_nc=1,
                output_nc=16,
                ngf=16,
                num_downs=4,)
    amix.load_state_dict(torch.load(f'/data/vision/polina/users/sebodiaz/projects/Act2Learn/pretrained/anatomix.pth'), strict=True)
    return torch.nn.Sequential(amix, torch.nn.InstanceNorm3d(16))
    

# Get optimizer
def get_optimizer(model, opts):
    import optimizers
    optimizer = optimizers.get_optimizer(model, opts)
    return optimizer

# Get scheduler
def get_scheduler(optimizer, opts, trainloader):
    import optimizers
    scheduler = optimizers.get_scheduler(optimizer, opts, trainloader)
    return scheduler

# Load model
def load_model(model, opts, optimizer=None, scheduler=None):
    # Load the model for resuming training or finetuning
    if opts.stage == 'train':
        checkpoint = torch.load(f'{opts.continue_path}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        continue_epoch = checkpoint['epoch'] + 1
        return model, optimizer, scheduler, continue_epoch
    
    if opts.stage == 'finetune':
        checkpoint = torch.load(f'{opts.continue_path}')
        model.load_state_dict(checkpoint['model'])
        return model, None, None, None
    
    # Load the model for testing
    elif opts.stage == 'test' or opts.stage == 'inference':
        checkpoint = torch.load(f'{opts.continue_path}')
        model.load_state_dict(checkpoint['model']) 
        return model

# 
def get_model(opts):
    import get_model

    return get_model.get_models(opts)

# Training function
def train(epoch, model, dataloader, loss_fn, optimizer, scheduler, scaler, fabric, opts, amix):
    # Set the model to training mode // important for batchnorm layers
    model.train()
    
    if amix is not None:
        amix.eval()
    
    # Intialize the cumulative loss // for logging purposes
    cumulative_loss = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', ncols=70, leave=False)
    progress_bar.set_description_str(f'Training | Epoch {epoch}')
    
    for i, (vol, heatmap, segmentation) in progress_bar:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=opts.use_amp):
            # perform the training steps
            optimizer.zero_grad()                                               # zero the gradients
            
            if amix is not None:
                with torch.no_grad():
                    feats = amix(vol)                                                # apply the amix model to the input volume
                    vol   = torch.cat((vol, feats), dim=1)                          # concatenate the amix features to the input volume
                    if opts.dropout > 0:
                        vol = torch.nn.functional.dropout3d(vol, p=opts.dropout, training=True)
            
            loss    = loss_fn(model, vol, [heatmap, segmentation], opts, 'train')
            fabric.backward(loss)                                               # backward pass
            optimizer.step()                                                    # update the weights
            
            # log the loss amd the learning rate to wandb and global step
            wandb.log({'Training/step_loss': loss.item()}); 
            wandb.log({'Misc/step': i + epoch * len(dataloader)});
            wandb.log({'Dynamics/learning_rate': optimizer.param_groups[0]['lr']})
            
            # update the progress bar
            progress_bar.set_description_str(f'Training | Epoch {epoch}')       # set the epoch in the progress bar
            progress_bar.set_postfix_str(f'Loss: {loss.item():.5f}')            # print the loss
            
            # update the cumulative loss
            cumulative_loss += loss.item()
            scheduler.step()                                                    # update the learning rate
        

            
    # Log the epoch loss to wandb
    cumulative_loss /= len(dataloader)

    wandb.log({'Training/epoch_loss': cumulative_loss, 'epoch': epoch})
        
    return

# Validation function
def validate(epoch, model, dataloader, loss_fn, scaler, opts, amix):
    """ Validate the model. We use PCK for keypoint regression and Dice for segmentation. """
    # Set the model to evaluation mode
    model.eval()
    
    if amix is not None:
        amix.eval()
        
    
    progress_bar    = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation', ncols=70, leave=False)
    progress_bar.set_description_str(f'Validation | Epoch {epoch}')

    
    pcks    = []
    haussdorffs   = []
    # Iterate over the validation dataset
    for ii, (vol, joint_coord, segmentations) in progress_bar:
        with torch.no_grad():
            if amix is not None:
                with torch.no_grad():
                    feats = amix(vol)                                                # apply the amix model to the input volume
                    vol   = torch.cat((vol, feats), dim=1)                          # concatenate the amix features to the input volume
            
            # Run model's forward pass
            prediction      = model(vol)
            
            # Get the individual keypoints
            joint_coord     = joint_coord[0].cpu().numpy()
            
            # If the model is trained with segmentation, get the segmentation output
            if opts.train_type == 'seg+pose':
                seg_pred        = prediction[:, 0:2, ...] # get the segmentation output
                prediction      = prediction[:, 2:, ...]  # get the keypoint output
                
                probs           = torch.softmax(seg_pred, dim=1)
                preds           = torch.argmax(probs, dim=1, keepdim=True)
                hd              = monai.metrics.compute_hausdorff_distance(include_background=False, y_pred=preds, y=segmentations)
                haussdorffs.append(hd.cpu().numpy())

                
                kp_pred         = torch.nn.functional.relu(prediction).cpu()[0].numpy() 

            elif opts.train_type == 'pose':
                kp_pred         = torch.nn.functional.relu(prediction).cpu()[0].numpy() # get the keypoint output
            
            elif opts.train_type == 'seg':
                probs          = torch.softmax(prediction, dim=1)
                preds          = torch.argmax(probs, dim=1, keepdim=True)
                hd             = monai.metrics.compute_hausdorff_distance(include_background=False, y_pred=preds, y=segmentations)
                haussdorffs.append(hd.cpu().numpy())
                
                


        if opts.train_type != 'seg':
            predict_coord = np.zeros_like(joint_coord)
            for i in range(joint_coord.shape[-1]):
                if joint_coord[2, i] <= 0:
                    joint_coord[:, i]   = np.nan
                    predict_coord[i, :] = np.nan
                else:
                    volume  = kp_pred[i]
                    ind     = np.unravel_index(np.argmax(volume), volume.shape)
                    weights = 1e-10
                    x_p = y_p = z_p = 0
                    for x in range(ind[1]-1, ind[1]+2):
                        for y in range(ind[0]-1, ind[0]+2):
                            for z in range(ind[2]-1, ind[2]+2):
                                if 0 <= x and x < volume.shape[1] and 0 <= y < volume.shape[0] and 0 <= z < volume.shape[2]:
                                    if volume[y, x, z] > weights:
                                        weights         += volume[y, x, z]
                                        x_p             += x * volume[y, x, z]
                                        y_p             += y * volume[y, x, z]
                                        z_p             += z * volume[y, x, z]
                    predict_coord[0, i] = x_p / weights + 1 # add 1 to the coordinates, this is for 1-based indexing; this is different from inference code, when we can choose
                    predict_coord[1, i] = y_p / weights + 1 # 
                    predict_coord[2, i] = z_p / weights + 1 #   
            
            e = predict_coord - joint_coord # shape (3, 15)
            e = (e[0,:]**2 + e[1,:]**2 + e[2,:]**2)**0.5 # shape (15,)
            pcks.append(e)
    if opts.train_type != 'seg':
        error                                       = np.array(pcks) # shape (len(testloader), 15)
        error_pck                                   = error.copy()
        error_pck[error > opts.error_threshold]     = 0 # allocate 0 if the error is greater than the threshold
        error_pck[error <= opts.error_threshold]    = 1 # allocate 1 if the error is less than the threshold
        pck_error                                   = np.sum(error_pck, 0) / error_pck.shape[0]
        pck_error_all                               = np.round(np.mean(pck_error), 3) # total average pck error across all joints
    
    if opts.train_type == 'seg+pose':
        # Log the validation loss and pck error to wandb
        segmentation_loss = np.mean(haussdorffs)
        wandb.log({'Validation/haussdorff_loss': segmentation_loss, 'Validation/pck_error': pck_error_all})
        
    
    elif opts.train_type == 'pose':
        # Log the validation loss and pck error to wandb
        wandb.log({'Validation/pck_error': pck_error_all})
        segmentation_loss = 0
    
    elif opts.train_type == 'seg':
        # Log the validation loss and pck error to wandb
        segmentation_loss = np.mean(haussdorffs)
        wandb.log({'Validation/haussdorff_loss': segmentation_loss})
        pck_error_all = 0
    
    
    
    
        
    return pck_error_all, segmentation_loss

# Test function
def test(model, dataloader, opts):
    # Set the model to evaluation // important for batchnorm layers
    model.eval()
    
    # Initialize the empty coordinate results
    res     = []
    error   = []
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Testing', ncols=70, leave=False)
    
    for i, (x, joint_coord, dn) in progress_bar:
        with torch.no_grad():
            dn          = dn.item() # get the dn
            x           = x.cuda() # move the input to gpu
            hm_         = model(x) # get the heatmap
            hm          = torch.nn.functional.relu(hm_).cpu()[0].numpy() #torch.nn.functional.relu(hm_).cpu()[0].numpy() # move the heatmap to cpu
            joint_coord = joint_coord[0].numpy() # get the joint coordinates

        # Gather the predicted coordinates based on argmax
        predict_coord = np.zeros_like(joint_coord) # initialize the predicted coordinates)
        mat_coords    = []
        
        ## get the predicted coordinates by looping over the joints   
        for i in range(joint_coord.shape[-1]):
            if joint_coord[2, i] <= 0:
                joint_coord[:, i]   = np.nan
                predict_coord[i, :] = np.nan
            else:
                volume  = hm[i]
                ind     = np.unravel_index(np.argmax(volume), volume.shape)
                weights = 1e-10
                x_p = y_p = z_p = 0
                for x in range(ind[1]-1, ind[1]+2):
                    for y in range(ind[0]-1, ind[0]+2):
                        for z in range(ind[2]-1, ind[2]+2):
                            if 0 <= x and x < volume.shape[1] and 0 <= y < volume.shape[0] and 0 <= z < volume.shape[2]:
                                if volume[y, x, z] > weights:
                                    weights         += volume[y, x, z]
                                    x_p             += x * volume[y, x, z]
                                    y_p             += y * volume[y, x, z]
                                    z_p             += z * volume[y, x, z]
                predict_coord[0, i] = x_p / weights + 1 # add 1 to the coordinates, this is for 1-based indexing; this is different from inference code, when we can choose
                predict_coord[1, i] = y_p / weights + 1 # 
                predict_coord[2, i] = z_p / weights + 1 #    
        #mat_coords.append(predict_coord)
        
        # calculate the error
        e = predict_coord - joint_coord # shape (3, 15)
        e = (e[0,:]**2 + e[1,:]**2 + e[2,:]**2)**0.5 # shape (15,)
        res.append(np.hstack((dn, predict_coord.ravel(), joint_coord.ravel())))
        error.append(e)

    # PCK error calculations
    error                                       = np.array(error) # shape (len(testloader), 15)
    joint_mean_error                            = np.mean(error, axis=0)
    mean_error                                  = np.round(np.mean(error), 2) # mean error across all volumes per joint
    error_pck                                   = error.copy()
    error_pck[error > opts.error_threshold]     = 0 # allocate 0 if the error is greater than the threshold
    error_pck[error <= opts.error_threshold]    = 1 # allocate 1 if the error is less than the threshold
    pck_error = np.sum(error_pck, 0) / error_pck.shape[0]
    pck_error_all = np.round(np.mean(pck_error), 3) # total average pck error across all joints

    # Save the pck error to a csv file
    np.savetxt(f'{opts.save_path}/{opts.run_name}_pck_{np.round(mean_error,3)}_{np.round(pck_error_all,3)}_thresh_{opts.error_threshold}.csv', pck_error, fmt='%.3f')
    
    # Save the average keypoint error to a csv file
    np.savetxt(f'{opts.save_path}/{opts.run_name}_jointmeanerror.csv', joint_mean_error, fmt='%.3f')

# Inference function
def inference(model, opts):
    # Set up directories
    mkdir(f'{opts.output_path}/{opts.run_name}')
    # set the model to evaluation // important for batchnorm layers
    model.eval()
    # intialize the empty coordinate results
    joint_coords = []

    nii_files = sorted([f for f in os.listdir(opts.rawdata_path) if f.endswith('.nii.gz')])

    # Iterate through the filtered files
    for idx, f in enumerate(tqdm(nii_files, ncols=70, total=len(nii_files), desc=f"Patient: {opts.label_name}")):
        # 
        final           = f
        
        # Load volumes using nibabel
        vol             = tio.ScalarImage(f'{opts.rawdata_path}/{f}')
        resolution      = vol.spacing
        factors         = np.array(resolution)
        vol             = vol.data[0]
        percentile_fac  = np.percentile(vol[vol > 0], 99)
        vol             = vol / percentile_fac

        # Pad the volume to be divisible by 2**depth
        pf          = 2**opts.depth
        pad_width   = [(0, ceil(s/pf)*pf-s) for s in vol.shape]
        volume      = np.pad(vol, pad_width, mode='constant')
        volume      = torch.Tensor(volume).unsqueeze(0).unsqueeze(0).cuda()
        
        # Get the resolution of the volume
        _, _, xx, yy, zz = volume.shape
        
        # Create the heatmap volume
        hm_volume  = np.zeros((opts.nJoints, xx, yy, zz))
                
        # Create empty array for joint coordinates
        joint_coord = np.zeros((3, opts.nJoints))

        with torch.no_grad():
            # Get the model output
            hm_ = model(volume)
            hm  = torch.nn.functional.relu(hm_).cpu()[0].numpy() # threshold the heatmap and move it to cpu
            # Get the frame
            f   = int(f[:4])
            
            # Loop over the keypoints
            for i in range(joint_coord.shape[-1]):
                # Get the heatmap
                kp  = hm[i]                        
                hm_volume[i] = kp
                
                # Get the predicted coordinates
                ind = np.unravel_index(np.argmax(kp), kp.shape)

                # Weight the predictions
                weights = 1e-10
                x_p = y_p = z_p = 0
                for x in range(ind[1]-1,ind[1]+2):
                    for y in range(ind[0]-1,ind[0]+2):
                        for z in range(ind[2]-1,ind[2]+2):
                            if 0 <= x < kp.shape[1] and 0 <= y < kp.shape[0] and 0 <= z < kp.shape[2]:
                                if kp[y, x, z] > weights:
                                    weights += kp[y, x, z]
                                    x_p += x * kp[y, x, z] # * resolution[1]
                                    y_p += y * kp[y, x, z] # * resolution[0]
                                    z_p += z * kp[y, x, z] # * resolution[2]
                joint_coord[0, i] = x_p / weights + opts.index_type # plus 1 for the 1-based indexing
                joint_coord[1, i] = y_p / weights + opts.index_type
                joint_coord[2, i] = z_p / weights + opts.index_type

            # Append the joint coordinates to the list
            joint_coords.append(joint_coord)

    io.savemat(f'{opts.output_path}/{opts.run_name}/{opts.label_name}.mat', {'joint_coord': np.stack(joint_coords), 'factors': factors})
    return
           
