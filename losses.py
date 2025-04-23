""""
This file contains the loss function for the model.

Originally wrriten by Sebo 01/07/2025

"""

import torch
import wandb
from monai.losses import LocalNormalizedCrossCorrelationLoss as LNCC
from monai.losses import DiceCELoss


import torch

import torch

class AdaptiveLoss(torch.nn.Module):
    def __init__(self, lambda_reg=0.1, eps=1e-6):
        """
        Adaptive loss combining heatmap regression and regularizer loss.

        Args:
            lambda_reg (float): Weight for the regularizer loss (default: 0.1).
            eps (float): Small constant for numerical stability.
        """
        super(AdaptiveLoss, self).__init__()
        self.lambda_reg = lambda_reg  # Regularization weight
        self.eps = eps  # Small constant to prevent division by zero

    def forward(self, output, target):
        """
        Compute the adaptive loss combining Eq. 8 and Eq. 9.

        Args:
            output (torch.Tensor): Predicted scale map (B, C, H, W).
            target (torch.Tensor): Ground-truth heatmap H_sigma0 (B, C, H, W).

        Returns:
            torch.Tensor: Total loss value.
        """
        # Generate binary mask where target heatmap is nonzero
        mask = (target > 0).float()

        # Clamp output to avoid division by zero or very small numbers
        output = torch.clamp(output, min=self.eps, max=10.0)  # Cap at 10 to prevent extreme values

        # Compute alpha = (1/s - 1), clamping to prevent extreme values
        alpha = torch.clamp((1 / output) - 1, min=-5, max=5)  # Keep within reasonable range

        # Compute log_target safely
        log_target = torch.where(target > 0, torch.log(torch.clamp(target, min=self.eps)), torch.zeros_like(target))

        # Equation 8: Taylor series approximation for scale-adaptive heatmap
        H_sigma0_s = torch.where(
            target > 0,
            0.5 * target * (1 + (1 + alpha * log_target) ** 2),  # Taylor expansion
            torch.zeros_like(target)
        )

        # Equation 9: Compute total loss
        regression_loss = torch.mean((output - H_sigma0_s) ** 2 * mask)  # Normalize loss scale
        regularizer_loss = torch.mean(alpha ** 2 * mask)  # Keep this in range

        total_loss = regression_loss + self.lambda_reg * regularizer_loss

        return total_loss

class SplitLoss(torch.nn.Module):
    def __init__(self, weight=0.01):
        """
        Adaptive loss combining heatmap regression and regularizer loss.

        Args:
            lambda_reg (float): Weight for the regularizer loss (default: 0.1).
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.weight = weight

    def forward(self, predictions, target):
        nKeypoints = predictions.shape[1] // 2
        mu         = predictions[:, :nKeypoints, ...]
        logvar     = predictions[:, nKeypoints:, ...]

        sigma2     = torch.exp(logvar).clamp(min=1e-6)
        mse        = ((target - mu) ** 2) / (2 * sigma2)
        log        = 0.5 * torch.log1p(sigma2)

        loss       = (mse + log).mean()

        var_penal  = self.weight * sigma2.mean()
        return loss + var_penal
# Define the loss function class
class Losses(torch.nn.Module):
    def __init__(self, opts):
        super().__init__()
        
        # Define the loss function
        if opts.loss == 'mse' or opts.loss == 'repulsive':
            print('Using MSE or repulsive loss')
            self.loss = torch.nn.MSELoss()      # MSE Loss, standard for heatmap regression
        elif opts.loss == 'l1':
            print('Using L1 Loss')
            self.loss = torch.nn.L1Loss()       # L1 Loss. Haven't tested yet.
        elif opts.loss == 'smooth_l1':
            print('Using Smooth L1 Loss')
            self.loss = torch.nn.SmoothL1Loss() # Smooth L1 Loss. Haven't tested yet.
        elif opts.loss == 'ncc':
            print('Using NCC Loss')             # NCC Loss. Haven't tested yet.
            self.l1_loss = torch.nn.L1Loss()
            self.ncc_loss = LNCC(kernel_size=5)
        elif opts.loss == 'dice_ce':
            print('Using DiceCE Loss')
            self.loss = DiceCELoss()
            self.loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
        elif opts.loss == 'adaptive':
            self.loss = AdaptiveLoss()
        elif opts.loss == 'split':
            self.loss = SplitLoss()
        
        self.nl = torch.nn.ReLU()
        self.sm = torch.nn.Softmax

    def apply_softmax_relu(self, heatmaps):
        """
        Apply Softmax and then ReLU to a batch of heatmaps (B, K, H, W, D).
        """
        # Apply Softmax to the spatial dimensions (H, W, D) across each keypoint (K)
        heatmaps = torch.nn.functional.softmax(heatmaps.view(heatmaps.shape[0], heatmaps.shape[1], -1), dim=-1).view(heatmaps.shape)
        # Apply ReLU to ensure non-negative values
        return torch.nn.functional.relu(heatmaps)

    def repulsion_loss(self, heatmaps):
        """
        Penalizes overlapping heatmaps by computing the element-wise product between different heatmaps.
        
        heatmaps: Tensor of shape (B, K, H, W, D), where K is the number of keypoints.
        Returns: Scalar loss term
        """
        B, K, H, W, D = heatmaps.shape
        loss = 0.0
        
        for i in range(K):
            for j in range(i + 1, K):  # Avoid duplicate pairs
                # Apply softmax and ReLU to each heatmap pair
                hm_i = self.apply_softmax_relu(heatmaps[:, i])
                hm_j = self.apply_softmax_relu(heatmaps[:, j])
                # Compute the overlap penalty between the heatmaps
                loss += (hm_i * hm_j).mean()

        # Normalize the loss by the batch size and number of keypoints
        return loss # / (B * K)
    
    def _soft_argmax(self, output, alpha=1000.0):
        assert output.dim()==5
        # alpha is here to make the largest element really big, so it
        # would become very close to 1 after softmax
        alpha = 1000.0 
        N,C,H,W,D = output.shape
        soft_max = torch.nn.functional.softmax(output.view(N,C,-1)*alpha,dim=2)
        soft_max = soft_max.view(output.shape)
        indices_kernel = torch.arange(start=0,end=H*W*D, device=output.device).unsqueeze(0)
        indices_kernel = indices_kernel.view((H,W,D))
        conv = soft_max*indices_kernel
        indices = conv.sum(2).sum(2).sum(2)
        z = indices%D
        y = (indices/D).floor()%W
        x = (((indices/D).floor())/W).floor()%H
        coords = torch.stack([x,y,z],dim=2)
        return coords
    
    def _joint_consistency(self, output, target, weight):
        # compute the soft argmax for joint consistency
        outputs = self._soft_argmax(output) # shape (B, 15, 3)
        target  = self._soft_argmax(target) # shape (B, 15, 3)
        
        # compute the length between channels 0 and 2
        l_shin    = torch.linalg.norm(outputs[:, 0, :] - outputs[:, 2, :], dim=1) # for left shin
        r_shin    = torch.linalg.norm(outputs[:, 1, :] - outputs[:, 3, :], dim=1) # for right shin
        l_thigh   = torch.linalg.norm(outputs[:, 2, :] - outputs[:, 9, :], dim=1) # for left thigh
        r_thigh   = torch.linalg.norm(outputs[:, 3, :] - outputs[:, 10, :], dim=1)# for right thigh
        l_forearm = torch.linalg.norm(outputs[:, 6, :] - outputs[:, 13, :], dim=1)# for left forearm
        r_forearm = torch.linalg.norm(outputs[:, 7, :] - outputs[:, 14, :], dim=1)# for right forearm
        l_bicep   = torch.linalg.norm(outputs[:, 6, :] - outputs[:, 11, :], dim=1)# for left bicep
        r_bicep   = torch.linalg.norm(outputs[:, 7, :] - outputs[:, 12, :], dim=1)# for right bicep
        
        # compute mean squared difference between left and right limbs
        jc_loss   = (l_shin - r_shin).abs() + (l_thigh - r_thigh).abs() + (l_forearm - r_forearm).abs() + (l_bicep - r_bicep).abs()
        jc_loss   = weight * torch.mean(jc_loss)
        wandb.log({'joint_consistency_loss': jc_loss.item()})
        return jc_loss
            
    def forward(self, model, data, target, opts, stage = 'train', mask=None):
        # Forward pass
        if opts.dsnt is True:
            coords, heatmaps = model(data)
            mse_loss = average_loss(euclidean(coords, target) + jenson_shannon(heatmaps, target, 2), mask)
        else:
            output      = model(data)
            if opts.loss == 'adaptive':
                loss    = self.loss(output, target)
            elif opts.unet_type == 'hourglass':
                losses = 0.0
                for out in output:
                    losses += self.loss(out, target)
                loss = losses / len(output)
            
            elif opts.loss == 'repulsive':
                loss = self.loss(output, target)
                repulse = self.repulsion_loss(output)
                loss = loss + 0.1 * repulse
            
            
            else:
                loss    = self.loss(output, target)
        
        # Log MSE and joint consistency separably to wandb
        if stage == 'train':
            wandb.log({'mse_loss': loss.item()})
                
        return loss
    
# Get the loss function
def get_loss_fn(opts):
    loss_fn = Losses(opts)
    return loss_fn

from functools import reduce
from operator import mul


"""

Below are some functions to recreate the DSNT network loss proposed by 

"""

# Euclidean distance // nothing fancy, but I mask the keypoints within the frame
def euclidean(actual, target):
    """Calculate the Euclidean losses for multi-point samples.

    Each sample must contain n points, each with d dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).

    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)


    Returns:
        Tensor: Losses (B x L)
    """
    #print(f'actual shape: {actual.shape}, target shape: {target.shape}')
    assert actual.size() == target.size(), 'input tensors must have the same size'
    return torch.norm(actual - target, p=2, dim=-1, keepdim=False)

def jenson_shannon(heatmaps, mu_t, sigma_t):
    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _js_div_3d)

def _divergence_reg_losses(heatmaps, mu_t, sigma_t, divergence):
    gauss = make_gauss(mu_t, heatmaps.size()[2:], sigma_t)
    div   = divergence(heatmaps, gauss)
    return div

def _kl_div_3d(p, q, eps=1e-8):
    # D_KL(P || Q)
    batch, chans, height, width, depth = p.shape
    unsummed_kl = torch.nn.functional.kl_div(
        (q + eps).reshape(batch * chans, height * width * depth).log(), p.reshape(batch * chans, height * width * depth), reduction="none"
    )
    kl_values = unsummed_kl.sum(-1).view(batch, chans)
    return kl_values

def _js_div_3d(p, q):
    # JSD(P || Q)
    m = 0.5 * (p + q)
    return 0.5 * _kl_div_3d(p, m) + 0.5 * _kl_div_3d(q, m)

def normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.

    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:

    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```

    Args:
        length: The length of the vector

    Returns:
        The generated vector
    """
    if isinstance(length, torch.Tensor):
        length = length.to(device, dtype)
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype, device=device) * (2.0 / length) + first

def make_gauss(means, size, sigma, normalize=True):
    """Draw Gaussians.

    This function is differential with respect to means.

    Note on ordering: `size` expects [..., depth, height, width], whereas
    `means` expects x, y, z, ...

    Args:
        means: coordinates containing the Gaussian means (units: normalized coordinates)
        size: size of the generated images (units: pixels)
        sigma: standard deviation of the Gaussian (units: pixels)
        normalize: when set to True, the returned Gaussians will be normalized
    """

    dim_range = range(-1, -(len(size) + 1), -1)
    coords_list = [normalized_linspace(s, dtype=means.dtype, device=means.device)
                   for s in reversed(size)]

    # PDF = exp(-(x - \mu)^2 / (2 \sigma^2))

    # dists <- (x - \mu)^2
    dists = [(x - mean) ** 2 for x, mean in zip(coords_list, means.split(1, -1))]

    # ks <- -1 / (2 \sigma^2)
    stddevs = [2 * sigma / s for s in reversed(size)]
    ks = [-0.5 * (1 / stddev) ** 2 for stddev in stddevs]

    exps = [(dist * k).exp() for k, dist in zip(ks, dists)]

    # Combine dimensions of the Gaussian
    gauss = reduce(mul, [
        reduce(lambda t, d: t.unsqueeze(d), filter(lambda d: d != dim, dim_range), dist)
        for dim, dist in zip(dim_range, exps)
    ])

    if not normalize:
        return gauss

    # Normalize the Gaussians
    val_sum = reduce(lambda t, dim: t.sum(dim, keepdim=True), dim_range, gauss) + 1e-24
    return gauss / val_sum


def average_loss(losses, mask=None):
    """Calculate the average of per-location losses.

    Args:
        losses (Tensor): Predictions (B x L)
        mask (Tensor, optional): Mask of points to include in the loss calculation
            (B x L), defaults to including everything
    """

    if mask is not None:
        assert mask.size() == losses.size(), 'mask must be the same size as losses'
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom

if __name__ == "__main__":
    # Sample from the standard normal distribution
    x = torch.randn(1, 15, 64, 64, 64)

    # Flatten the spatial dimensions (the last three dimensions)
    x_flat = x.view(1, 15, -1)

    # Apply softmax along the flattened spatial dimension so that the values sum to 1
    x_softmax = torch.nn.functional.softmax(x_flat, dim=-1)

    # Reshape back to the original shape
    x_softmax = x_softmax.view(1, 15, 64, 64, 64)                   # sum to 1 (shape: 3, 15, 64, 64, 64)
    samples = 2 * torch.rand(1,15,3) - 1  # samples uniformly in [-1, 1]
    js = jenson_shannon(heatmaps=x_softmax,
                    mu_t=samples,
                    sigma_t=2)
    
    eu = euclidean(torch.randn(1, 15, 3), torch.randn(1, 15, 3))
    mas = torch.randn(1, 15) > 0.5

    loss = average_loss(eu + js, mas)

    print(loss)