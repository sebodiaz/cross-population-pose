"""

Main file for the project. This the file called by the bash script.

Originally wrriten by Sebo 01/07/2025.

"""

# Import the necessary libraries
import utils
import losses
import optimizers
import options
import wandb
import torch
import lightning
import numpy as np

# enable Tensor cores // only set if this is applicable to your hardware
# BE mindful of this... ! 
torch.set_float32_matmul_precision('high')

# define global step
global_step = 0

# define the main function
def main(opts):
    # Get the model
    model = utils.get_model(opts)
    
    # Setup Fabric (useful for distributed training & mixed precision)
    fabric = lightning.fabric(
        accelerator="gpu",
        devices=opts.num_gpus,
        num_nodes=opts.num_nodes,
        precision="16-mixed" if opts.use_amp else "32"
    )
    fabric.launch()

    # -------------------- Train or Finetune Stage --------------------
    if opts.stage in ['train', 'finetune']:
        
        # Get and setup dataloaders
        trainloader, valloader = utils.get_dataloader(opts)
        trainloader, valloader = fabric.setup_dataloaders(trainloader, valloader)

        # Define loss, optimizer, scheduler
        loss_fn    = losses.get_loss_fn(opts)
        loss_fn    = fabric.setup(loss_fn)
        optimizer  = optimizers.get_optimizer(model, opts)
        scheduler  = optimizers.get_scheduler(optimizer, opts, trainloader)
        scaler     = torch.cuda.amp.GradScaler(enabled=opts.use_amp)

        # Setup model and optimizer with Fabric
        model, optimizer = fabric.setup(model, optimizer)
        
        # Load checkpoint if continuing or finetuning
        if opts.continue_path is not None:
            if opts.stage == 'train':
                model, optimizer, scheduler, continue_epoch = utils.load_model(model, opts, optimizer, scheduler)
                start_epoch = continue_epoch
            elif opts.stage == 'finetune':
                model, _, _, _ = utils.load_model(model, opts, optimizer, scheduler)
                start_epoch = 0
        else:
            start_epoch = 0

        # Initialize best validation score (PCK)
        best_val = float('-inf')  # Higher is better for PCK

        # -------------------- Training Loop --------------------
        for epoch in range(start_epoch, opts.epochs):
            utils.train(epoch, model, trainloader, loss_fn, optimizer, scheduler, fabric, opts)
            utils.save_model_latest(model, optimizer, scheduler, epoch, opts)

            # Run validation at specified frequency or final epoch
            if (epoch % opts.val_freq == 0 or epoch == opts.epochs - 1) and epoch > 0:
                if opts.stage == 'train':
                    pck = utils.validate(epoch, model, valloader, opts)
                    if pck > best_val:
                        best_val = pck
                        utils.save_model_best(model, optimizer, scheduler, epoch, opts)

        wandb.finish()  # End wandb run

    # -------------------- Test Stage --------------------
    elif opts.stage == 'test':
        testloader  = utils.get_offline_dataloader(opts)
        model       = utils.load_model(model, opts)
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




