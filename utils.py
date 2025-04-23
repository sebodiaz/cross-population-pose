"""
Utility functions.

Originally wrriten by Sebo 01/07/2025. Some functions are copy/pasted from Molin Zhang and Junshen Xu.

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
import nibabel as nib
from math import ceil
import scipy.io as io
import random
import skimage
import scipy
import online
import torchio as tio
from lightning.fabric import Fabric

def resample_volume(volume, original_resolution, target_resolution=(3, 3, 3)):
    """
    Resample a volume to a target resolution.
    
    Parameters:
    - volume: 3D numpy array
    - original_resolution: tuple of original voxel sizes
    - target_resolution: tuple of target voxel sizes (default 3mm)
    
    Returns:
    - Resampled volume
    """
    # Calculate zoom factors
    zoom_factors = tuple(orig / target for orig, target in 
                         zip(original_resolution, target_resolution))
    #print(zoom_factors)
    # Resample the volume
    resampled_volume = scipy.ndimage.zoom(volume, zoom_factors, order=3)  # Linear interpolation
    
    return resampled_volume, zoom_factors

# Define some utility functions
def check_arg(parse_fn=None, valid_list=None):
    def new_parse_fn(x):
        if parse_fn is not None:
            x = parse_fn(x)
        if valid_list is not None:
            if x not in valid_list:
                raise ValueError()
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

def print_argsv2(args):
    args = to_dict(args)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

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

# Visualize the predictions
def log_visualize(input_volume, pred_volume, target_volume, epoch):
    """
    Create side-by-side scrollable slice visualizations for WandB.
    Similar to matplotlib's scroll viewer but using Plotly for WandB compatibility.
    
    Args:
        pred_volume (torch.Tensor): Predicted volume (1, 15, 64, 64, 64)
        target_volume (torch.Tensor): Target volume (1, 15, 64, 64, 64)
        step (int): Current step/epoch for W&B logging
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Move tensors to CPU and convert to numpy
    pred = pred_volume.detach().cpu().numpy()[0]     # (15, 64, 64, 64)
    target = target_volume.detach().cpu().numpy()[0]
    vol = input_volume.detach().cpu().numpy()[0, 0]  # Assuming input_volume has shape (1, 15, 64, 64, 64)
    
    # Normalize input, prediction, and target volumes
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))  # Normalize input volume
    
    # Channel dictionary
    channels = {
        0: 'left ankle',
        1: 'right ankle',
        2: 'left knee',
        3: 'right knee',
        4: 'bladder',
        5: 'left elbow',
        6: 'right elbow',
        7: 'left eye',
        8: 'right eye',
        9: 'left hip',
        10: 'right hip',
        11: 'left shoulder',
        12: 'right shoulder',
        13: 'left wrist',
        14: 'right wrist'
    }

    # Create a figure for each channel
    for channel in range(15):
        # Create subplot with three side-by-side plots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(f'Input - {channels[channel]}', f'Prediction - {channels[channel]}', f'Ground Truth - {channels[channel]}'),
            horizontal_spacing=0.05
        )
        
        # Add input volume slice (middle slice in the input volume)
        fig.add_trace(
            go.Heatmap(
                z=vol[:, :, vol.shape[-1]//2],  # Middle slice of the input volume for the current channel
                colorscale='gray',
                showscale=True,
                name='Input'
            ),
            row=1, col=1
        )
        
        # Add prediction volume slice (middle slice in the predicted volume)
        fig.add_trace(
            go.Heatmap(
                z=pred[channel, :, :, pred.shape[-1]//2],
                colorscale='viridis',
                showscale=True,
                name='Prediction'
            ),
            row=1, col=2
        )
        
        # Add target volume slice (middle slice in the target volume)
        fig.add_trace(
            go.Heatmap(
                z=target[channel, :, :, target.shape[-1]//2],
                colorscale='viridis',
                showscale=True,
                name='Ground Truth'
            ),
            row=1, col=3
        )
        
        # Create frames for each slice (animation frames)
        frames = []
        for slice_idx in range(pred.shape[-1]):
            frame = go.Frame(
                data=[
                    # Input slice for the given frame
                    go.Heatmap(z=vol[:, :, slice_idx], colorscale='gray'),
                    go.Heatmap(z=pred[channel, :, :, slice_idx], colorscale='viridis'),
                    go.Heatmap(z=target[channel, :, :, slice_idx], colorscale='viridis')
                ],
                name=str(slice_idx)
            )
            frames.append(frame)
        
        # Add slider
        sliders = [{
            'active': pred.shape[-1]//2,  # Start from middle slice
            'currentvalue': {
                'prefix': 'Slice: ',
                'visible': True,
                'xanchor': 'center'
            },
            'transition': {'duration': 0},
            'steps': [
                {
                    'args': [
                        [str(i)],
                        {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ],
                    'label': str(i),
                    'method': 'animate'
                } for i in range(pred.shape[-1])
            ],
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'pad': {'b': 10, 't': 50}
        }]
        
        # Update layout
        fig.update_layout(
            title=f'{channels[channel]}',
            width=1500,  # Increased width for 3 subplots
            height=500,
            sliders=sliders,
            showlegend=False
        )
        
        # Ensure both plots have the same scale
        max_val = max(pred[channel].max(), target[channel].max())
        min_val = min(pred[channel].min(), target[channel].min())
        
        fig.update_traces(
            zmin=min_val,
            zmax=max_val
        )
        
        # Add frames to figure
        fig.frames = frames
        
        # Log to WandB
        wandb.log({
            f'Epoch_{epoch}_{channels[channel]}': wandb.Html(fig.to_html(include_plotlyjs='cdn')),
        })


    """
    Create side-by-side scrollable slice visualizations for WandB.
    Similar to matplotlib's scroll viewer but using Plotly for WandB compatibility.
    
    Args:
        pred_volume (torch.Tensor): Predicted volume (1, 15, 64, 64, 64)
        target_volume (torch.Tensor): Target volume (1, 15, 64, 64, 64)
        input_volume (torch.Tensor): Input volume (1, 15, 64, 64, 64)
        epoch (int): Current step/epoch for W&B logging
    """
    # Move tensors to CPU and convert to numpy
    input_vol = input_volume.detach().cpu().numpy()[0]  # (15, 64, 64, 64)
    pred = pred_volume.detach().cpu().numpy()[0]  # (15, 64, 64, 64)
    target = target_volume.detach().cpu().numpy()[0] # (1, 64, 64, 64)

    # ReLU the predictions
    pred = np.maximum(pred, 0)

    # Normalize
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    target = (target - np.min(target)) / (np.max(target) - np.min(target))
    #input_vol = (input_vol - np.min(input_vol)) / (np.max(input_vol) - np.min(input_vol))

    # Channel dictionary
    channels = {
        0: 'left ankle',
        1: 'right ankle',
        2: 'left knee',
        3: 'right knee',
        4: 'bladder',
        5: 'left elbow',
        6: 'right elbow',
        7: 'left eye',
        8: 'right eye',
        9: 'left hip',
        10: 'right hip',
        11: 'left shoulder',
        12: 'right shoulder',
        13: 'left wrist',
        14: 'right wrist'
    }

    # Create a figure for each channel
    for channel in range(15):
        # Create subplot with three side-by-side plots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(f'Input - {channels[channel]}', f'Prediction - {channels[channel]}', f'Ground Truth - {channels[channel]}'),
            horizontal_spacing=0.05
        )

        # Add input volume heatmap
        fig.add_trace(
            go.Heatmap(
                z=input_vol[:, :, input_vol.shape[-1]//2],
                colorscale='gray',
                showscale=True,
                name='Input'
            ),
            row=1, col=1
        )

        # Add predicted volume heatmap
        fig.add_trace(
            go.Heatmap(
                z=pred[channel, :, :, pred.shape[-1]//2],
                colorscale='viridis',
                showscale=True,
                name='Prediction'
            ),
            row=1, col=2
        )

        # Add ground truth volume heatmap
        fig.add_trace(
            go.Heatmap(
                z=target[channel, :, :, target.shape[-1]//2],
                colorscale='viridis',
                showscale=True,
                name='Ground Truth'
            ),
            row=1, col=3
        )

        # Create frames for each slice
        frames = []
        for slice_idx in range(pred.shape[-1]):
            frame = go.Frame(
                data=[
                    go.Heatmap(z=input_vol[:, :, slice_idx], colorscale='gray'),
                    go.Heatmap(z=pred[channel, :, :, slice_idx], colorscale='viridis'),
                    go.Heatmap(z=target[channel, :, :, slice_idx], colorscale='viridis')
                ],
                name=str(slice_idx)
            )
            frames.append(frame)

        # Add slider
        sliders = [{
            'active': pred.shape[-1] // 2,  # Start from middle slice
            'currentvalue': {
                'prefix': 'Slice: ',
                'visible': True,
                'xanchor': 'center'
            },
            'transition': {'duration': 0},
            'steps': [
                {
                    'args': [
                        [str(i)],
                        {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ],
                    'label': str(i),
                    'method': 'animate'
                } for i in range(pred.shape[-1])
            ],
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'pad': {'b': 10, 't': 50}
        }]

        # Update layout
        fig.update_layout(
            title=f'{channels[channel]}',
            width=1200,
            height=500,
            sliders=sliders,
            showlegend=False
        )
        # Add frames to figure
        fig.frames = frames

        # Log to WandB
        wandb.log({
            f'Epoch_{epoch}_{channels[channel]}': wandb.Html(fig.to_html(include_plotlyjs='cdn'))
        })
        
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
        val_dl      = DataLoader(train_val, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

        return train_dl, val_dl
    
    elif opts.stage == 'finetune':
        # create the dataset
        train_ds    = data.FinalDataset('finetune', opts)
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
    }, f'{opts.save_path}/checkpoints/best.pth')

def save_model_latest(model, optimizer, scheduler, epoch, opts):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, f'{opts.save_path}/checkpoints/latest.pth')

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



def get_model(opts):
    import get_model

    return get_model.get_models(opts)

# Get the model
def get_model_old(opts):
    # Load the necessary library
    import models

    if opts.anatomix is True:
        amix_model = models.Unet(dimension=3, input_nc=1, output_nc=16, ngf=16, num_downs=4).cuda()
        amix_model.load_state_dict(torch.load('/data/vision/polina/users/sebodiaz/data/anatomix.pth'))
        fin_layer = torch.nn.InstanceNorm3d(16).cuda()
        amix = torch.nn.Sequential(amix_model, fin_layer)
        print(f'Anatomix model loaded.')
    else:
        amix = None

    # More efficient Unet
    if opts.vit is True:
        model = models.TokenPose(opts)#.cuda()
        print('Loaded ViT network.')
    else:
        if opts.unet_type == 'small' and opts.dsnt is False and opts.anatomix is False and opts.tsm is False and opts.four_dim is False and opts.middle_prediction is False:
            model = models.Unet(dimension=3, input_nc=opts.temporal, output_nc=opts.nJoints * opts.temporal, ngf=opts.nFeat // 4, num_downs=opts.depth + 1)#.cuda()
            print(f'Loaded simple concatenation model for temporal == {opts.temporal}')
        elif opts.unet_type == 'small' and opts.dsnt is False and opts.anatomix is False and opts.tsm is True and opts.four_dim is False and opts.middle_prediction is True:
            model = models.TemporalShiftUnet(dimension=3, input_nc=1, output_nc=opts.nJoints, ngf=opts.nFeat // 4, num_downs=opts.depth + 1, n_segment=opts.temporal, tsm_levels=[0,1,2], middle_prediction=True)#.cuda()
        elif opts.unet_type == 'small' and opts.dsnt is False and opts.anatomix is False and opts.tsm is True and opts.four_dim is False:
            print(f'Loading TSM UNet.')
            model = models.TemporalShiftUnet(dimension=3, input_nc=1, output_nc=opts.nJoints, ngf=opts.nFeat // 4, num_downs=opts.depth + 1, n_segment=opts.temporal, tsm_levels=[0,1,2])#.cuda()
        elif opts.unet_type == 'small' and opts.dsnt is False and opts.anatomix is False and opts.tsm is False and opts.four_dim is True:
            model = models.Unet(dimension=4, input_nc=1, output_nc=opts.nJoints, num_downs=opts.depth, ngf=opts.nFeat // 4, pad_type="zeros")#.cuda()
            print('Loaded 4D-UNet.')
        elif opts.unet_type == 'small' and opts.dsnt is False and opts.anatomix is True:
            print(f"Loading Anatomix UNet.")
            model = models.Unet(dimension=3, input_nc=1+16, output_nc=opts.nJoints, ngf=opts.nFeat // 4, num_downs=opts.depth + 1).cuda()
        elif opts.unet_type == 'small' and opts.dsnt is True:
            model = models.DSNT(opts, n_locations=15)#.cuda()
        # Bigger Unet // what was originally used by Junshen
        elif opts.unet_type == 'big':
            print(f"Loading Original Junshen UNet.")
            model = models.UNet3D(1, opts.nFeat, opts.nJoints, opts.depth, opts.use_bias, opts.norm_layer)#.cuda()
            
    return model, amix
    

# Training function
def train(epoch, model, dataloader, loss_fn, optimizer, scheduler, scaler, amix, fabric, opts):
    # Set the model to training mode // important for batchnorm layers
    model.train()
    if opts.anatomix is True: amix.eval()
    
    # Intialize the cumulative loss // for logging purposes
    cumulative_loss = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', ncols=70, leave=False)
    progress_bar.set_description_str(f'Training | Epoch {epoch}')
    
    for i, (vol, target) in progress_bar:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=opts.use_amp):
            # perform the training steps
            optimizer.zero_grad()                                           # zero the gradients
            if opts.anatomix is True:
                with torch.no_grad():
                    amix_vol = amix(vol)
                    if opts.concat:
                        vol      = torch.cat((vol, amix_vol), 1)
                    else:
                        vol      = amix_vol
            loss    = loss_fn(model, vol, target, opts, 'train')
            fabric.backward(loss)                                               # backward pass
            optimizer.step()                                                    # update the weights
            
            # log the loss amd the learning rate to wandb and global step
            wandb.log({'train_step_loss': loss.item()}); 
            wandb.log({'step': i + len(dataloader) * epoch});
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})
            
            # update the progress bar
            progress_bar.set_description_str(f'Training | Epoch {epoch}')       # set the epoch in the progress bar
            progress_bar.set_postfix_str(f'Loss: {loss.item():.5f}')            # print the loss
            
            # update the cumulative loss
            cumulative_loss += loss.item()
            scheduler.step()                                                    # update the learning rate
        

            
    # Log the epoch loss to wandb
    if opts.train_type == 'online':
        cumulative_loss /= opts.steps_per_epoch
    elif opts.train_type == 'offline':
        cumulative_loss /= len(dataloader)

    wandb.log({'epoch_loss': cumulative_loss, 'epoch': epoch})
        
    return

# Validation function
def validate(epoch, model, dataloader, loss_fn, scaler, amix, opts):
    # Set the model to evaluation mode
    model.eval()
    if opts.anatomix is True:
        amix.eval()
    
    progress_bar    = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation', ncols=70, leave=False)
    progress_bar.set_description_str(f'Validation | Epoch {epoch}')
    cumulative_loss = 0
    
    # Iterate over the validation dataset
    for i, (vol, target) in progress_bar:
        # Perform the validation steps
        with torch.no_grad():
            if opts.anatomix is True:
                amix_vol = amix(vol)
                if opts.concat:
                    vol = torch.cat((vol, amix_vol), 1)
                else:
                    vol = amix_vol
            loss = loss_fn(model, vol, target, opts, 'val')     # calculate the loss
            
        # Update the progress bar
        progress_bar.set_description_str(f'Validation | Epoch {epoch}')          # set the epoch in the progress bar
        progress_bar.set_postfix_str(f'Loss: {loss.item():.4f}')    # print the loss
        
        # Update the cumulative loss
        cumulative_loss += loss.item()
        
    # Log the epoch loss to wandb
    cumulative_loss /= len(dataloader)
    wandb.log({'val_loss': cumulative_loss})
    return cumulative_loss

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
def inference(model, amix, opts):
    # Set up directories
    mkdir(f'{opts.output_path}/{opts.run_name}')
    # set the model to evaluation // important for batchnorm layers
    model.eval()
    if opts.anatomix:
        amix.eval()
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
            if opts.anatomix:
                output = amix(volume)
                volume = torch.cat((volume, output), 1)
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

    if opts.output_vol is True:
        # Load the reference image and segmentation
        dat = nib.load(f'{opts.rawdata_path}/{final}')
        dat_data = dat.get_fdata().copy()  # Create a copy of the original data
        original_shape = dat_data.shape
        
        # Loop through the volumes and fill in the data
        jcs = np.stack(joint_coords)
        for i in range(len(jcs)):
            # Create a new volume with EXACTLY the same dimensions as the original
            empty_vol = np.zeros_like(dat_data)
            
            for j in range(opts.nJoints):
                x, y, z = jcs[i][:, j]
                
                # Check if coordinates are within bounds before assigning
                if (0 <= int(y) < original_shape[0] and 
                    0 <= int(x) < original_shape[1] and 
                    0 <= int(z) < original_shape[2]):
                    empty_vol[int(y), int(x), int(z)] = int(1 + j)
                else:
                    print(f"Warning: Joint {j} at position ({x},{y},{z}) is outside volume boundaries")
                    # Optionally clip to boundaries:
                    # clipped_y = max(0, min(int(y), original_shape[0]-1))
                    # clipped_x = max(0, min(int(x), original_shape[1]-1))
                    # clipped_z = max(0, min(int(z), original_shape[2]-1))
                    # empty_vol[clipped_y, clipped_x, clipped_z] = int(1 + j)
            
            # Create a new NIfTI with EXACTLY the same affine as the original
            nii = nib.Nifti1Image(empty_vol, dat.affine)
            
            # Copy header information from the original (this preserves additional metadata)
            for key in dat.header:
                if key != 'dim' and key != 'pixdim':  # Don't copy dimension info as it's set by the array
                    nii.header[key] = dat.header[key]
            
            # Save the segmentation with the original volume properties
            istr = str(i).zfill(4)
            nib.save(nii, f'{opts.output_path}/{opts.run_name}/{istr}.nii.gz')


    return
 
def inference_old(model, amix, opts):
    # Set up directories
    mkdir(f'{opts.output_path}/{opts.run_name}')
    if opts.output_vis is True:
        mkdir(f'{opts.output_path}/{opts.label_name}/heatmaps')
        mkdir(f'{opts.output_path}/{opts.label_name}/volumes')
    
    # set the model to evaluation // important for batchnorm layers
    model.eval()
    
    # intialize the empty coordinate results
    joint_coords = []
    
    # if top-k is implemented, initialize the empty arrays
    if opts.top_k is not None:
        all_joints   = np.zeros((len(os.listdir(opts.rawdata_path)), opts.top_k, 3, opts.nJoints))
        all_values   = np.zeros((len(os.listdir(opts.rawdata_path)), opts.top_k, opts.nJoints))
    hm_volumes   = []
    nii_files = sorted([f for f in os.listdir(opts.rawdata_path) if f.endswith('.nii.gz')])

    # Iterate through the filtered files
    for idx, f in enumerate(tqdm(nii_files, ncols=70, total=len(nii_files), desc=f"Patient: {opts.label_name}")):
        final = f
        # Load volumes using nibabel
        vol         = tio.ScalarImage(f'{opts.rawdata_path}/{f}')
        resolution  = vol.spacing
        factors     = np.array(resolution)
        
        # Resample if the resolution is not 3mm
        if opts.resample is True and resolution != (3, 3, 3):
            vol     = tio.Resample((3, 3, 3))(vol)
        vol = vol.data[0]
        
        # Normalize the data by 99th percentile
        percentile_fac = np.percentile(vol[vol > 0], 99)
        vol = vol / percentile_fac
        
        if opts.unet_type == 'small':
            depth = opts.depth + 1
        else:
            depth = opts.depth

        # Pad the volume to be divisible by 2**depth
        pf          = 2**depth
        pad_width   = [(0, ceil(s/pf)*pf-s) for s in vol.shape]
        volume      = np.pad(vol, pad_width, mode='constant')
        volume      = torch.Tensor(volume).unsqueeze(0).unsqueeze(0).cuda()

        if opts.anatomix is True:
            with torch.no_grad():
                amix_vol = amix(volume)
                volume   = torch.cat((volume, amix_vol), 1)

        # Get the resolution of the volume
        _, _, xx, yy, zz = volume.shape
        
        # Create the heatmap volume
        hm_volume  = np.zeros((opts.nJoints, xx, yy, zz))
                
        # Create empty array for joint coordinates
        joint_coord = np.zeros((3, opts.nJoints))
        
        # Run the predictions
        ms = []
        cs = []
        with torch.no_grad():
            # Get the model output
            hm_ = model(volume)
            hm  = torch.nn.functional.relu(hm_).cpu()[0].numpy() # threshold the heatmap and move it to cpu
            means_peaks, covars = fitGaussian(hm)
            ms.append(means_peaks)
            cs.append(covars)
            # Get the frame
            f   = int(f[:4])
            
            # Loop over the keypoints
            for i in range(joint_coord.shape[-1]):
                # Get the heatmap
                kp  = hm[i]
                
                # if top-k is implemented, get the local maxima
                if opts.top_k is not None:
                    # get the local maxima
                    maxima = skimage.feature.peak_local_max(kp, min_distance=10, num_peaks=opts.top_k) # shape (top_k, 3)
                    all_joints[idx, :, :, i] = maxima
                    
                    # get confidence scores for the top-k local maxima
                    for kk in range(opts.top_k):
                        xs, ys, zs = maxima[kk]
                        all_values[idx, kk, i] = kp[xs, ys, zs]
                        
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

            # Append the heatmap volume to the list
            hm_volumes.append(hm_volume)
        
        if opts.output_vis is True:
            # Save the heatmap and volume *.mat files
            idx = str(idx).zfill(4)
            io.savemat(f'{opts.output_path}/{opts.label_name}/heatmaps/{idx}.mat', {'heatmap': hm_volume})
            io.savemat(f'{opts.output_path}/{opts.label_name}/volumes/{idx}.mat', {'volume': mat_volume, 'resolution': resolution})

        
    # Assertions to ensure the correct number of joint coordinates and the correct shape
    #assert len(joint_coords) == len(os.listdir(opts.rawdata_path)), f"Expected {len(os.listdir(opts.rawdata_path))} joint coordinates, got {len(joint_coords)}"
    #assert joint_coords[0].shape == (3, opts.nJoints), f"Expected shape (3, {opts.nJoints}), got {joint_coords[0].shape}"
    
    # Save the joint coordinates to a *.mat file
    if opts.top_k is not None:
        io.savemat(f'{opts.output_path}/{opts.label_name}/{opts.label_name}.mat', {'joint_coord': np.stack(joint_coords), 'all_joints': all_joints, 'all_values': all_values})
    else:
        io.savemat(f'{opts.output_path}/{opts.run_name}/{opts.label_name}.mat', {'joint_coord': np.stack(joint_coords), 'factors': factors})  

    if opts.output_vol is True:
        # Load the reference image and segmentation
        dat = nib.load(f'{opts.rawdata_path}/{final}')
        dat_data = dat.get_fdata().copy()  # Create a copy of the original data
        original_shape = dat_data.shape
        
        # Loop through the volumes and fill in the data
        jcs = np.stack(joint_coords)
        for i in range(len(jcs)):
            # Create a new volume with EXACTLY the same dimensions as the original
            empty_vol = np.zeros_like(dat_data)
            
            for j in range(opts.nJoints):
                x, y, z = jcs[i][:, j]
                
                # Check if coordinates are within bounds before assigning
                if (0 <= int(y) < original_shape[0] and 
                    0 <= int(x) < original_shape[1] and 
                    0 <= int(z) < original_shape[2]):
                    empty_vol[int(y), int(x), int(z)] = int(1 + j)
                else:
                    print(f"Warning: Joint {j} at position ({x},{y},{z}) is outside volume boundaries")
                    # Optionally clip to boundaries:
                    # clipped_y = max(0, min(int(y), original_shape[0]-1))
                    # clipped_x = max(0, min(int(x), original_shape[1]-1))
                    # clipped_z = max(0, min(int(z), original_shape[2]-1))
                    # empty_vol[clipped_y, clipped_x, clipped_z] = int(1 + j)
            
            # Create a new NIfTI with EXACTLY the same affine as the original
            nii = nib.Nifti1Image(empty_vol, dat.affine)
            
            # Copy header information from the original (this preserves additional metadata)
            for key in dat.header:
                if key != 'dim' and key != 'pixdim':  # Don't copy dimension info as it's set by the array
                    nii.header[key] = dat.header[key]
            
            # Save the segmentation with the original volume properties
            istr = str(i).zfill(4)
            nib.save(nii, f'{opts.output_path}/{opts.run_name}/{istr}.nii.gz')
    
    if opts.output_vol is False:
        # make heatmaps dir
        mkdir(f'{opts.output_path}/{opts.run_name}/heatmaps')
        # stakc the heatmap volumes
        hm_volumes = np.stack(hm_volumes)

        # save first 10 volumes as npz
        np.savez_compressed(f'{opts.output_path}/{opts.run_name}/heatmaps.npz', (hm_volumes[:25]).astype(np.float32))  

        print(f'Heatmap volumes shape: {hm_volumes.shape}')
        for t in range(hm_volumes.shape[0]):
            hm_vol = hm_volumes[t]
            hm_vol = hm_vol.transpose(1, 2, 3, 0)
            hm_vol = nib.Nifti1Image(hm_vol, dat.affine)
            nib.save(hm_vol, f'{opts.output_path}/{opts.run_name}/heatmaps/{t}.nii.gz')

    # stack the ms and cs
    io.savemat(f'{opts.output_path}/{opts.run_name}/peaks.mat', {'means': np.stack(ms), 'covars': np.stack(cs)})
    return

def inference_temporal(model, opts):
    # determine whether to condition on the entire sequence

    # Set up directories
    mkdir(f'{opts.output_path}/{opts.run_name}')
    
    # set the model to evaluation // important for batchnorm layers
    model.eval()
    
    # Intialize the empty coordinate results
    joint_coords = []
    
    # Setup the files
    nii_files = sorted([f for f in os.listdir(opts.rawdata_path) if f.endswith('.nii.gz')])
    
    # Pad the volume to be divisible by 2**depth
    pf          = 2**opts.depth
    
    # Iterate through the filtered files
    for idx, f in enumerate(tqdm(nii_files, ncols=70, total=len(nii_files), desc=f"Patient: {opts.label_name}")):
        
        # Loop through each window opts.temporal times
        hms = []
        for w in range(opts.condition):
            # Load volumes using nibabel
            vols = []
   
            # Handle temporal windows with wraparound
            if opts.condition  == 1:
                for ii in range(opts.temporal):
                    # Calculate the index with wraparound
                    frame_idx = (w + idx + ii - opts.temporal + 1) % len(nii_files)  # This will wrap around at the start
                    first_part, second_part = nii_files[frame_idx].split("_", 1)
                    second_part             = second_part.split(".")[0]  # Remove the extension
                    ff = first_part + '_' + second_part + ".nii.gz"
                    vol = tio.ScalarImage(f'{opts.rawdata_path}/{ff}')
                    resolution = vol.spacing
                    factors = np.array(resolution)
                    vol = vol.data[0]
                    vol = vol / np.percentile(vol[vol > 0], 99)
                    pad_width = [(0, ceil(s / pf) * pf - s) for s in vol.shape]
                    volume = np.pad(vol, pad_width, mode='constant')
                    volume = torch.Tensor(volume).unsqueeze(0).unsqueeze(0).cuda()
                    _, _, xx, yy, zz = volume.shape
                    vols.append(volume.permute(1,0,2,3,4))
            else:
                possible_indices = list(range(len(nii_files)))
                possible_indices.remove(idx)  # Exclude current index

                # Select (opts.temporal - 1) unique random files
                rand_indices = random.sample(possible_indices, opts.temporal - 1)

                # Process the current file
                first_part, second_part = f.split("_", 1)
                second_part = second_part.split(".")[0]
                ff_current = first_part + '_' + second_part + ".nii.gz"
                current_vol = tio.ScalarImage(f'{opts.rawdata_path}/{ff_current}')
                
                current_data = current_vol.data[0]
                current_data = current_data / np.percentile(current_data[current_data > 0], 99)
                pad_width_current = [(0, ceil(s / pf) * pf - s) for s in current_data.shape]
                current_volume = np.pad(current_data, pad_width_current, mode='constant')
                current_volume = torch.Tensor(current_volume).unsqueeze(0).unsqueeze(0).cuda()

                # Collect volumes
                temporal_stack = [current_volume.permute(1, 0, 2, 3, 4)]

                # Process the randomly selected files
                for rand_idx in rand_indices:
                    random_file = nii_files[rand_idx]
                    first_part_rand, second_part_rand = random_file.split("_", 1)
                    second_part_rand = second_part_rand.split(".")[0]
                    ff_rand = first_part_rand + '_' + second_part_rand + ".nii.gz"
                    random_vol = tio.ScalarImage(f'{opts.rawdata_path}/{ff_rand}')
                    
                    random_data = random_vol.data[0]
                    random_data = random_data / np.percentile(random_data[random_data > 0], 99)
                    pad_width_rand = [(0, ceil(s / pf) * pf - s) for s in random_data.shape]
                    random_volume = np.pad(random_data, pad_width_rand, mode='constant')
                    random_volume = torch.Tensor(random_volume).unsqueeze(0).unsqueeze(0).cuda()
                    
                    # Append to the temporal stack
                    temporal_stack.append(random_volume.permute(1, 0, 2, 3, 4))

                # Concatenate along the temporal dimension
                vols.append(torch.cat(temporal_stack, dim=0))
            vols = torch.stack(vols, axis=1)
            # Create empty array for joint coordinates
            joint_coord = np.zeros((3, opts.nJoints))
            
            
            # Run the predictions
            with torch.no_grad():
                # Get the model output
                hm_ = model(vols) # outputted shape should be (1, opts.temporal, 15, H, W, D)
                ijx = -1-w if opts.condition == 1 else 0
                hm  = torch.nn.functional.relu(hm_).cpu()[0, ijx, ...].numpy() # threshold the heatmap and move it to cpu
                hms.append(hm)
        hms = np.stack(hms)
        hm = np.mean(hms, axis=0)
        # Get the frame
        f   = int(f[:4])
        
        # Loop over the keypoints
        for i in range(joint_coord.shape[-1]):
            # Get the heatmap
            kp  = hm[i]
    
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
        
    # Save the joint coordinates to a *.mat file
    io.savemat(f'{opts.output_path}/{opts.run_name}/{opts.label_name}.mat', {'joint_coord': np.stack(joint_coords), 'factors': factors})  

    return



import skimage
def fitGaussian(heatmaps, k=3, local_size=6, min_distance=7):
    nJoints, H, W, D = heatmaps.shape
    heatmaps_nps     = heatmaps#.cpu().numpy()
    means            = np.zeros((nJoints, k, 3))
    covariances      = np.zeros((nJoints, k, 3, 3))
    peak_values      = np.zeros((nJoints, k))  # To store peak values
    hsize            = local_size // 2
    
    for kp in range(nJoints):
        # Find local maxima
        coords       = skimage.feature.peak_local_max(heatmaps[kp], min_distance=min_distance, num_peaks=k)
        means[kp]    = coords
        
        for j, (z, y, x) in enumerate(coords):
            # Extract the local region coordinates centered around the max activation
            z_min, z_max = max(0, z - hsize), min(H, z + hsize + 1)
            y_min, y_max = max(0, y - hsize), min(H, y + hsize + 1)
            x_min, x_max = max(0, x - hsize), min(W, x + hsize + 1)  # Ensure x_max is within bounds
            
            # Get the local region and the peak value at the coordinate
            local_path   = heatmaps_nps[kp, z_min:z_max, y_min:y_max, x_min:x_max]
            peak_values[kp, j] = np.max(local_path)  # Store the peak value
            
            # Compute the mean and covariance of the local region
            local_coords = np.array(np.meshgrid(
                np.arange(z_min, z_max),
                np.arange(y_min, y_max),
                np.arange(x_min, x_max),
                indexing='ij')).reshape(3, -1).T

            # Local values
            local_values = local_path.flatten()
            local_mean   = np.average(local_coords, axis=0, weights=local_values)
            diffs        = local_coords - local_mean
            covariance   = np.cov(diffs.T, aweights=local_values)
            covariances[kp, j] = np.nan_to_num(covariance)

    # Combine means and peak_values into a single array
    means_and_peaks = np.concatenate([means, peak_values[..., np.newaxis]], axis=-1)  # Shape: (nJoints, k, 4)

    return means_and_peaks, covariances


            
            
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