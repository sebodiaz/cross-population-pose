"""

Main file for the project. This the file called by the bash script.

Originally wrriten by Sebo 01/07/2025

"""

# Import the necessary libraries
import utils
import losses
import optimizers
import options
import wandb
import torch
from lightning.fabric import Fabric # this is for multi-gpu 
import numpy as np

# Enable Tensor core for A6000 and RTX6000
torch.set_float32_matmul_precision('high')

# Define global step
global_step = 0

# Define the main function
def main(opts):
    
    # Get the model and amix, if applicable
    model = utils.get_model(opts)
    
    # Setup Fabric
    fabric      = Fabric(accelerator="gpu", devices=opts.num_gpus, num_nodes=opts.num_nodes, precision="16-mixed" if opts.use_amp else "32")
    fabric.launch()
    
    # Use amix features if applicable
    if opts.use_amix:
        amix = utils.get_amix(opts)
        amix = fabric.setup(amix)
    else:
        amix = None


    # Train stage
    if opts.stage in ['train', 'finetune']:
        
        # Get the dataloaders
        trainloader, valloader = utils.get_dataloaders(opts)
        trainloader, valloader = fabric.setup_dataloaders(trainloader, valloader)

        # Define the loss function, optimizer, and scheduler
        loss_fn                = losses.get_loss_fn(opts)
        loss_fn                = fabric.setup(loss_fn)
        optimizer              = optimizers.get_optimizer(model, loss_fn, opts)
        scheduler              = optimizers.get_scheduler(optimizer, opts, trainloader)
        scaler                 = torch.cuda.amp.GradScaler(enabled=opts.use_amp)

        # Fabric setup
        model, optimizer       = fabric.setup(model, optimizer)
        
        
        # Initialize from checkpoint and set the start epoch, if applicable
        if opts.continue_path is not None and opts.stage == 'train':
            model, optimizer, scheduler, continue_epoch = utils.load_model(model, opts, optimizer, scheduler)
            start_epoch     = continue_epoch
        elif opts.continue_path is not None and opts.stage == 'finetune':
            model, _, _, _  = utils.load_model(model, opts, optimizer, scheduler)
            start_epoch     = 0
        else:
            
            start_epoch     = 0
        
        # Intialize the best val
        best_val        = float('-inf')
        
        # Loop over the epochs
        for epoch in range(start_epoch, opts.epochs):
            # Train function
            utils.train(epoch, model, trainloader, loss_fn, optimizer, scheduler, scaler, fabric, opts, amix) 
            
            # Save the latest model
            utils.save_model_latest(model, optimizer, scheduler, epoch, opts)

            # Validate the model every opts.val_freq epochs
            if (epoch % opts.val_freq == 0 or epoch == opts.epochs-1) and epoch > 0:
                if opts.stage == 'train':
                    # Validation function
                    pck, hausdorff  = utils.validate(epoch, model, valloader, loss_fn, scaler, opts, amix)
                    
                    # Log the metrics
                    utils.save_model_best(model, optimizer, scheduler, epoch, opts)
                
        # Finish wandb
        wandb.finish()
    
    # Test stage
    elif opts.stage == 'test':
        # Get the dataloader
        testloader = utils.get_offline_dataloader(opts)
        
        # Load the model
        model = utils.load_model(model, opts)
        
        # Test function
        utils.test(model, testloader, opts)
    
    # Inference stage
    elif opts.stage == 'inference':
        # Load the model
        model = utils.load_model(model, opts)
        model = fabric.setup(model)
        
        # Inference function
        utils.inference(model, opts) if opts.temporal < 2 else utils.inference_temporal(model, opts)



if __name__ == "__main__":
    # Get the options
    opts = options.Options().parse()
    
    # Call the main function
    main(opts)




