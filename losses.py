""""
This file contains the loss function for the model.

Originally wrriten by Sebo 01/07/2025.

"""

import torch
import wandb

class Losses(torch.nn.Module):
    def __init__(self):
        """
        
        Loss function class. Useful to implement other loss functions in the future.
        Currrently, it only contains heatmap regression (MSE) loss.
        
        
        """

        super().__init__()
        
        # Define the loss function
        self.loss = KeypointLoss()

    
    def forward(self, model, data, targets, opts, stage = 'train', mask=None):
        return self.loss(model, data, targets, stage=stage)

class KeypointLoss(torch.nn.Module):
    """ Loss function for keypoint regression."""
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
    
    def forward(self, model, data, targets, stage='train'):
        # Forward pass
        output       = model(data)
        loss         = self.loss(output, targets[0])
        
        # Log MSE and joint consistency separably to wandb
        if stage == 'train':
            wandb.log({'Training/reg_loss': loss.item()})
                
        return loss

# Get the loss function
def get_loss_fn(opts):
    return Losses()


if __name__ == "__main__":
    pass