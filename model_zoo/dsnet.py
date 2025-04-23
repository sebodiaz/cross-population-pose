import torch
from typing import Optional, Tuple
import small_unet

# Utility: Validate that the input is a 5D volume tensor.
def _validate_batched_volume_tensor_input(tensor: torch.Tensor) -> None:
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 5D tensor [B, C, D, H, W], but got {tensor.ndim} dimensions.")

def spatial_softmax3d(input: torch.Tensor, temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the softmax over the spatial dimensions of a 5D tensor.

    Args:
        input (Tensor): A tensor of shape [B, C, D, H, W].
        temperature (Optional[Tensor]): A temperature scaling factor (default: 1.0).

    Returns:
        Tensor: The spatially softmaxed tensor of shape [B, C, D, H, W].
    """
    _validate_batched_volume_tensor_input(input)
    if temperature is None:
        temperature = 1.0
    b, c, d, h, w = input.shape
    # Flatten the spatial dimensions (D*H*W) and apply softmax.
    input_reshaped = input.view(b, c, -1)
    softmax = F.softmax(input_reshaped / temperature, dim=-1)
    return softmax.view(b, c, d, h, w)

def spatial_expectation3d(input: torch.Tensor, normalized_coordinates: bool = True) -> torch.Tensor:
    r"""Compute a differentiable soft-argmax over a 5D tensor.

    This function computes the expected coordinate in each spatial dimension by
    marginalizing over the other two. By default the coordinate values are normalized
    so that the center of each cell lies within (-1, 1).

    Args:
        input (Tensor): A tensor of shape [B, C, D, H, W].
        normalized_coordinates (bool): If True, the coordinate ranges are [-1, 1].

    Returns:
        Tensor: Expected coordinates of shape [B, C, 3]. In Kornia style the coordinates
                are reversed so that the order is (x, y, z) where x corresponds to width.
    """
    _validate_batched_volume_tensor_input(input)
    b, c, d, h, w = input.shape
    if normalized_coordinates:
        # Create coordinate vectors corresponding to the center of each cell.
        d_range = torch.linspace(-1.0 + 1.0 / d, 1.0 - 1.0 / d, d, device=input.device, dtype=input.dtype)
        h_range = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=input.device, dtype=input.dtype)
        w_range = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=input.device, dtype=input.dtype)
    else:
        d_range = torch.arange(0, d, device=input.device, dtype=input.dtype)
        h_range = torch.arange(0, h, device=input.device, dtype=input.dtype)
        w_range = torch.arange(0, w, device=input.device, dtype=input.dtype)

    # Marginalize probabilities to compute the expectation in each spatial dimension.
    # Expectation along depth: sum_{d} (d_range * p(d))
    p_d = input.sum(dim=(3, 4))  # shape: [B, C, D]
    exp_d = (p_d * d_range.view(1, 1, d)).sum(dim=-1)  # [B, C]

    # Expectation along height: sum_{h} (h_range * p(h))
    p_h = input.sum(dim=(2, 4))  # shape: [B, C, H]
    exp_h = (p_h * h_range.view(1, 1, h)).sum(dim=-1)  # [B, C]

    # Expectation along width: sum_{w} (w_range * p(w))
    p_w = input.sum(dim=(2, 3))  # shape: [B, C, W]
    exp_w = (p_w * w_range.view(1, 1, w)).sum(dim=-1)  # [B, C]

    # Stack the expectations to form coordinates. This yields (depth, height, width).
    coords = torch.stack([exp_d, exp_h, exp_w], dim=-1)  # shape: [B, C, 3]
    # In Kornia the convention is to reverse the order so that the output is (x, y, z)
    coords = torch.cat(tuple(reversed(coords.split(1, dim=-1))), dim=-1)
    return coords

def _safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-32) -> torch.Tensor:
    r"""Safely perform element-wise division, clamping the denominator to avoid division by zero."""
    return numerator / torch.clamp(denominator, min=eps)

def render_gaussian3d(mean: torch.Tensor, std: torch.Tensor, size: Tuple[int, int, int], normalized_coordinates: bool = True) -> torch.Tensor:
    r"""Render a 3D Gaussian heatmap.

    Args:
        mean (Tensor): The center(s) of the Gaussian(s). Expected shape [B, 3] or [B, C, 3].
        std (Tensor): The standard deviation(s), with the same shape as `mean`.
        size (tuple[int, int, int]): The desired size (D, H, W) of the heatmap.
        normalized_coordinates (bool): Whether the coordinates are normalized to [-1, 1].

    Returns:
        Tensor: A 3D Gaussian heatmap of shape [B, 1, D, H, W] (or [B, C, D, H, W] if multiple channels).
    """
    d, h, w = size
    if normalized_coordinates:
        d_range = torch.linspace(-1.0 + 1.0 / d, 1.0 - 1.0 / d, d, device=mean.device, dtype=mean.dtype)
        h_range = torch.linspace(-1.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=mean.device, dtype=mean.dtype)
        w_range = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=mean.device, dtype=mean.dtype)
    else:
        d_range = torch.arange(0, d, device=mean.device, dtype=mean.dtype)
        h_range = torch.arange(0, h, device=mean.device, dtype=mean.dtype)
        w_range = torch.arange(0, w, device=mean.device, dtype=mean.dtype)
    
    # Create a meshgrid for the 3D coordinates.
    zz, yy, xx = torch.meshgrid(d_range, h_range, w_range, indexing='ij')
    grid = torch.stack([zz, yy, xx], dim=0)  # shape: [3, D, H, W]
    grid = grid.unsqueeze(0)  # add batch dimension: [1, 3, D, H, W]
    
    # Reshape mean and std to allow broadcasting.
    if mean.ndim == 2:  # [B, 3]
        mean = mean.view(mean.size(0), 3, 1, 1, 1)
        std = std.view(std.size(0), 3, 1, 1, 1)
    elif mean.ndim == 3:  # [B, C, 3]
        mean = mean.view(mean.size(0), mean.size(1), 3, 1, 1, 1)
        std = std.view(std.size(0), std.size(1), 3, 1, 1, 1)
        grid = grid.unsqueeze(1)  # shape becomes: [B, 1, 3, D, H, W]
    else:
        raise ValueError("`mean` must be a 2D or 3D tensor")
    
    # Compute the Gaussian: exponent = -0.5 * sum(((grid - mean) / std)²)
    if mean.ndim == 5:  # [B, 3, 1, 1, 1]
        gauss = torch.exp(-0.5 * (((grid - mean) / std) ** 2).sum(dim=1, keepdim=True))
    else:  # [B, C, 3, 1, 1, 1]
        gauss = torch.exp(-0.5 * (((grid - mean) / std) ** 2).sum(dim=2, keepdim=True))
    return gauss

# DSNT Module in Kornia style for 5D inputs.
class DSNT(torch.nn.Module):
    r"""Differentiable Spatial to Numerical Transform (DSNT) module for 3D volumes.

    This module uses a 3D fully convolutional network (e.g. a 3D UNet) to produce heatmaps.
    The heatmaps are normalized via a spatial softmax and then converted to numerical coordinates
    using a differentiable soft-argmax (i.e. spatial_expectation3d).

    Note:
        The UNet (or other backbone) must output a 5D tensor [B, C, D, H, W].
    """
    def __init__(self, opts, n_locations: int = 1):
        super().__init__()
        # Assuming that a 3D UNet implementation is available.
        self.fcn = small_unet.Unet(dimension=3,
                        input_nc=1,
                        output_nc=opts.nJoints,
                        ngf=opts.nFeat // 4,
                        num_downs=opts.depth + 1)# ).cuda()
        self.hm_conv = torch.nn.Conv3d(opts.nJoints, n_locations, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): Input tensor of shape [B, 1, D, H, W].

        Returns:
            Tuple[Tensor, Tensor]:
                - coords: Differentiable numerical coordinates, shape [B, n_locations, 3]
                          (ordered as [x, y, z], where x corresponds to the width axis).
                - heatmaps: Softmax-normalized heatmaps, shape [B, n_locations, D, H, W].
        """
        fnc_out = self.fcn(x)
        unnormalized_hms = self.hm_conv(fnc_out)
        heatmaps = spatial_softmax3d(unnormalized_hms)
        coords = spatial_expectation3d(heatmaps, normalized_coordinates=True)
        return coords, heatmaps
