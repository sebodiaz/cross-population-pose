"""
File containing the optimizer and scheduler functions.

Separate file from utils.py to keep it organized.

Originally wrriten by #### 01/07/2025
"""

import torch
import math

# Get the optimizer for the model 
def get_optimizer(model, loss_fn, opts):
    """
    Get the optimizer for the model based on the options.
    
    Args:
        model: The model to optimize
        opts: The options dictionary
        
    Returns:
        The optimizer for the model
    """
    if opts.learn_coeff:
        params_to_optimize = list(model.parameters()) + list(loss_fn.parameters())
    else:
        params_to_optimize = model.parameters()
    
    
    if opts.optimizer == 'adam': 
        optimizer = torch.optim.Adam(params_to_optimize,
                                     lr=opts.lr,
                                     weight_decay=opts.weight_decay)
    elif opts.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_to_optimize,
                                      lr=opts.lr,
                                      weight_decay=opts.weight_decay)
    
    

    return optimizer


# Define a custom Linear Warmup + Cosine Decay scheduler // Haven't tested yet, but is more modern
class LinearWarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0.0, last_epoch=-1):
        self.warmup_iters   = warmup_iters
        self.total_iters    = total_iters
        self.min_lr         = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter    = self.last_epoch + 1
        base_lrs        = self.base_lrs  # Initial learning rates set in optimizer
        
        if current_iter < self.warmup_iters:
            # Linear Warmup
            scale = current_iter / self.warmup_iters
            return [lr * scale for lr in base_lrs]
        else:
            # Cosine Decay
            decay_iter = current_iter - self.warmup_iters
            total_decay_iters = self.total_iters - self.warmup_iters
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_iter / total_decay_iters))
            return [self.min_lr + (lr - self.min_lr) * cosine_decay for lr in base_lrs]


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
    warmup_iters        = int(opts.warmup_epochs * len(trainloader))

    
    # Select the scheduler
    if opts.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, int(opts.lr_decay_ep * iter_per_epoch), eta_min=0, T_mult=2, last_epoch=-1
        )
    elif opts.lr_scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, 1.0, total_iters=total_iters
        )
    elif opts.lr_scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters
        )
    elif opts.lr_scheduler == 'cosinewarmup':
        scheduler = LinearWarmupCosineDecayLR(
            optimizer, warmup_iters=warmup_iters, total_iters=total_iters, min_lr=0.0
        )
    elif opts.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=opts.lr, total_steps=total_iters, pct_start=opts.warmup_epochs/opts.epochs
        )

    return scheduler
