""""
This file contains the loss function for the model.

Originally wrriten by Sebo 01/07/2025

"""

import torch
import wandb
import monai

class Losses(torch.nn.Module):
    def __init__(self, opts):
        super().__init__()
        
        # Define the loss function
        if opts.train_type == 'seg':
            self.loss = SegmentationLoss()
        elif opts.train_type == 'pose':
            self.loss = KeypointLoss()
        elif opts.train_type == 'seg+pose':
            self.loss = JointLoss(opts)
    
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

class SegmentationLoss(torch.nn.Module):
    """Loss function for segmentation."""
    def __init__(self):
        super().__init__()
        self.loss = monai.losses.DiceCELoss(
            include_background=False, to_onehot_y=True, softmax=True
        )
    
    def forward(self, model, data, targets, stage='train'):
        # Forward pass
        output       = model(data)
        loss         = self.loss(output, targets[1])
        
        # Log MSE and joint consistency separably to wandb
        if stage == 'train':
            wandb.log({'Training/seg_loss': loss.item()})
                
        return loss

class JointLoss(torch.nn.Module):
    """Loss function for joint training of segmentation and regression,
    with optional uncertainty-based dynamic weighting."""
    
    def __init__(self, opts):
        super().__init__()
        
        self.learn_coeff = opts.learn_coeff
        self.seg_loss_fn = monai.losses.DiceCELoss(
            include_background=False, to_onehot_y=True, softmax=True
        )
        self.reg_loss_fn = torch.nn.MSELoss()

        if self.learn_coeff:
            # Learnable log variances for uncertainty weighting
            self.log_sigma_seg = torch.nn.Parameter(torch.tensor(0.0))
            self.log_sigma_reg = torch.nn.Parameter(torch.tensor(0.0))
        else:
            # Static coefficients
            self.seg_coeff = opts.seg_coeff
            self.reg_coeff = opts.reg_coeff

    def forward(self, model, data, targets, stage='train'):
        output = model(data)

        seg_output = output[:, 0:2, :, :]
        reg_output = output[:, 2:, :, :]

        reg_loss = self.reg_loss_fn(reg_output, targets[0])
        seg_loss = self.seg_loss_fn(seg_output, targets[1])

        if self.learn_coeff:
            precision_seg = torch.exp(-2 * self.log_sigma_seg)
            precision_reg = torch.exp(-2 * self.log_sigma_reg)

            total_loss = (
                precision_seg * seg_loss +
                precision_reg * reg_loss +
                self.log_sigma_seg + self.log_sigma_reg
            )
        else:
            total_loss = self.seg_coeff * seg_loss + self.reg_coeff * reg_loss

        if stage == 'train':
            log_dict = {
                'Training/seg_loss': seg_loss.item(),
                'Training/reg_loss': reg_loss.item(),
            }
            if self.learn_coeff:
                log_dict.update({
                    'Training/log_sigma_seg': self.log_sigma_seg.item(),
                    'Training/log_sigma_reg': self.log_sigma_reg.item()
                })
            wandb.log(log_dict)

        return total_loss
    
# Get the loss function
def get_loss_fn(opts):
    loss_fn = Losses(opts)
    return loss_fn


if __name__ == "__main__":
    pass