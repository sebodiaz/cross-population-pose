"""
File containing the optimizer and scheduler functions.

Separate file from utils.py to keep it organized.

Originally wrriten by #### 01/07/2025
"""

import torch

# Get the optimizer for the model 
def get_optimizer(model, opts):
    """
    Get the optimizer for the model based on the options.
    
    Args:
        model: The model to optimize
        opts: The options dictionary
        
    Returns:
        The optimizer for the model
    """    
    if opts.optimizer == 'adam': 
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opts.lr,
                                     weight_decay=opts.weight_decay)
    elif opts.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=opts.lr,
                                      weight_decay=opts.weight_decay)
    
    return optimizer





def get_scheduler(optimizer, opts, trainloader):
    """
    Get the scheduler for the optimizer based on the options.
    
    Args:
        optimizer: The optimizer to schedule
        opts: The options dictionary
        trainloader: The data loader for training (used for step calculations)
        
    Returns:
        The scheduler for the optimizer
    """
    # Get necessary information
    total_iters         = int(opts.epochs * len(trainloader))
    iter_per_epoch      = len(trainloader)

    # Select the scheduler
    if opts.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, int(opts.lr_decay_ep * iter_per_epoch), eta_min=0, T_mult=2, last_epoch=-1
        )
    elif opts.lr_scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters
        )

    return scheduler
