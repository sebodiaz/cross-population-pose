import torch
from math import ceil
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union
import math

def max_pool4d(input, kernel_size: Union[int, tuple], stride: Union[int, tuple]):
    """
    Implements MaxPool4d (generalization of max_pool3d from PyTorch).

    Args:
        input (Tensor[N, C, K, D, H, W]): The input tensor or 6-dimensions with the first one being its batch i.e. a batch with ``N`` rows.
        kernel_size (tuple): Size of the kernel.
        stride (tuple): Stride of the kernel.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride, stride)
    kk, kd, kh, kw = kernel_size
    dk, dd, dh, dw = stride

    # get all image windows of size (kk, kd, kh, kw) and stride (dk, dd, dh, dw)
    input_windows = input.unfold(2, kk, dk).unfold(3, kd, dd).unfold(4, kh, dh).unfold(5, kw, dw)

    # view the windows as (kk * kd * kh * kw)
    input_windows = input_windows.contiguous().view(*input_windows.size()[:-4], -1)

    max_val, max_idx = input_windows.max(-1)
    
    return max_val, max_idx

def interpolate(input, size, mode="nearest", scale_factor=None):
    assert input.ndim >= 3
    if scale_factor is not None:
        raise NotImplementedError
    output_shape = (*input.shape[:2], *size)
    assert len(input.shape) == len(output_shape)
    # Apply linear interpolation to each spatial dimension.
    for i in range(2, 2 + len(size)):
        input_tail = math.prod(input.shape[i + 1 :])
        input = F.interpolate(
            input.reshape(
                input.shape[0], math.prod(input.shape[1:i]), input.shape[i], input_tail
            ),
            size=(output_shape[i], input_tail),
            mode=mode,
        ).reshape(*input.shape[:i], output_shape[i], *input.shape[i + 1 :])
    return input.reshape(output_shape)

class Upsampling6d(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale = scale_factor
        self.mode = mode
    
    def forward(self, x):
        B, T, C, H, W, D = x.shape
        return interpolate(input=x, size=(C, H*self.scale, W*self.scale, D*self.scale), mode=self.mode)


if __name__ == "__main__":
    x = torch.randn(4, 3, 4, 64, 64, 64)
    y = max_pool4d(x, 2, 2); print(y[0].shape)
    Upsample = Upsampling6d(scale_factor=2)
    interp = Upsample(y[0])
    print(interp.shape)

    x = torch.randn(4, 4, 32, 32, 32)
    y = torch.nn.Upsample(scale_factor=2, mode="nearest")(x)
    print(y.shape)