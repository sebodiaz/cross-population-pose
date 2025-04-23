import torch
import typing
import math
import einops

# Max pooling from https://github.com/FirasBDarwish/PyTorch4D/blob/main/src/torch4d/torch4d.py
def max_pool4d(input, kernel_size: typing.Union[int, tuple], stride: typing.Union[int, tuple]):
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
        input = torch.nn.functional.interpolate(
            input.reshape(
                input.shape[0], math.prod(input.shape[1:i]), input.shape[i], input_tail
            ),
            size=(output_shape[i], input_tail),
            mode=mode,
        ).reshape(*input.shape[:i], output_shape[i], *input.shape[i + 1 :])
    return input.reshape(output_shape)

def get_norm_layer(ndims, norm='batch'):
    """
    Get the normalization layer based on the number of dimensions and type of
    normalization.

    Parameters
    ----------
    ndims : int
        The number of dimensions for the normalization layer (1--3).
    norm : str, optional
        The type of normalization to use. 
        Options are 'batch', 'instance', or 'none'. 
        Default is 'batch'.

    Returns
    -------
    Norm : torch.nn.Module or None
        The corresponding PyTorch normalization layer, or None if 'none'.
    """
    if norm == 'batch':
        Norm = getattr(torch.nn, 'BatchNorm%dd' % ndims)
    elif norm == 'instance':
        Norm = getattr(torch.nn, 'InstanceNorm%dd' % ndims)
    elif norm == 'none':
        Norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return Norm

def get_actvn_layer(activation='relu'):
    """
    Get the activation function layer based on the provided activation type.

    Parameters
    ----------
    activation : str, optional
        The type of activation function to use. 
        Options are 'relu', 'lrelu', 'elu',  'prelu', 'selu', 'tanh', or 'none'
        Default is 'relu'.

    Returns
    -------
    Activation : torch.nn.Module or None
        The corresponding PyTorch activation layer, or None if 'none'.
    """

    if activation == 'relu':
        Activation = torch.nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        Activation = torch.nn.LeakyReLU(0.3, inplace=True)
    elif activation == 'elu':
        Activation = torch.nn.ELU()
    elif activation == 'prelu':
        Activation = torch.nn.PReLU()
    elif activation == 'selu':
        Activation = torch.nn.SELU(inplace=True)
    elif activation == 'tanh':
        Activation = torch.nn.Tanh()
    elif activation == 'none':
        Activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
    return Activation

def collapse_time(x):
    return einops.rearrange(x, 'B T C H W D -> (B T) C H W D')

def unravel_time(x, batch_size):
    return einops.rearrange(x, '(B T) C H W D -> B T C H W D', B=batch_size)


## Necessary classes for "SeboNet"
class MaxPool4d(torch.nn.Module):
    """
    See :func:`max_pool4d`.
    """

    def __init__(self, kernel_size: typing.Union[int, tuple], stride: typing.Union[int, tuple]) -> None:
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (Tensor): Input feature maps on which the maxpool operation will be applied (input assumed to be of shape: Tensor[N, C, K, D, H, W])
        
        Returns:
            returned_tensor (Tensor): The tensor after max pool operation is executed.
        """
        returned_tensor, indices = max_pool4d(input, self.kernel_size, self.stride)
        return returned_tensor

class Upsampling6D(torch.nn.Module):
    def __init__(self, scale_factor=2, mode="nearest", batch_size=4):
        super().__init__()
        self.scale = scale_factor
        self.mode = mode
        self.bs = batch_size
    
    def forward(self, x):
        #print(f'Prior to upsampling: {x.shape}')
        x = unravel_time(x, self.bs)
        #print(f'After unraveling: {x.shape}')

        B, T, C, H, W, D = x.shape
        x = interpolate(input=x, size=(C, H*self.scale, W*self.scale, D*self.scale), mode=self.mode)
        x = collapse_time(x)
        #print(f'After to upsampling: {x.shape}')
        return x

class TSM(torch.nn.Module):
    def __init__(self, net, n_segment=3, n_div=3, inplace=False, mode="residual"):
        super(TSM, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.mode = mode
    
    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)
    
    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, mode="residual"):
        
        nt, c, h, w, d  = x.size()
        bs              = nt // n_segment
        x               = einops.rearrange(x, '(B N) C H W D -> B N C H W D', B=nt//n_segment, N=n_segment)
        fold            = c // fold_div

        if mode == "residual":
            out                     = torch.zeros_like(x)
            out[:, :-1, :fold]      = x[:, 1:, :fold] - x[:, :-1, :fold]
            out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold] - x[:, 1:, fold:2*fold]
            out                     = x + out
            return einops.rearrange(out, 'B N C H W D -> (B N) C H W D')

class ConvBlock(torch.nn.Module):
    """
    A convolutional block with optional normalization and activation.

    This block performs a convolution, followed by optional normalization 
    and activation. The block supports 1D, 2D, and 3D convolutions.

    Parameters
    ----------
    ndims : int
        Number of dimensions (1, 2, or 3) for the convolution.
    input_dim : int
        Number of channels in the input.
    output_dim : int
        Number of channels in the output.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple
        Stride of the convolution.
    bias : bool
        Whether to use a bias term in the convolution.
    padding : int or tuple, optional
        Amount of padding to add to the input, by default 0.
    norm : str, optional
        Type of normalization to apply ('batch', 'instance', or 'none'), 
        by default 'none'.
    activation : str, optional
        Activation function to use ('relu', 'lrelu', 'elu', 'prelu', 'selu', 
        'tanh', or 'none'), by default 'relu'.
    pad_type : str, optional
        Type of padding to use ('zeros', 'reflect', etc.), by default 'zeros'.

    """

    def __init__(
        self, ndims, input_dim, output_dim, kernel_size, stride, use_bias,
        padding=0, norm='none', activation='relu', pad_type='zeros', tsm=False,
    ):
        """
        Initialize the ConvBlock with convolution, normalization, and 
        activation layers.

        Parameters are described in the class docstring.
        """
        super(ConvBlock, self).__init__()
        self.use_bias = use_bias
        assert ndims in [1, 2, 3, 4], 'ndims in 1--4. found: %d' % ndims # changed to 4D
        Conv = getattr(torch.nn, 'Conv%dd' % ndims)

        # initialize convolution
        self.conv = Conv(
            input_dim, output_dim, kernel_size, stride, bias=self.use_bias,
            padding=padding, padding_mode=pad_type
        )

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = getattr(torch.nn, 'BatchNorm%dd'%ndims)(norm_dim)
        elif norm == 'instance':
            self.norm = getattr(
                torch.nn, 'InstanceNorm%dd'%ndims
            )(norm_dim, track_running_stats=False)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = torch.nn.ELU()
        elif activation == 'prelu':
            self.activation = torch.nn.PReLU()
        elif activation == 'selu':
            self.activation = torch.nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        
        if tsm:
            self.tsm = TSM(torch.nn.Identity(), n_segment=4, n_div=4)
        else:
            self.tsm = None


    def forward(self, x):
        """
        Perform the forward pass through the ConvBlock.

        Applies convolution, followed by optional normalization and 
        activation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the block.

        Returns
        -------
        torch.Tensor
            The output tensor after applying convolution, normalization, 
            and activation.
        """
        if self.tsm:
            res = x
            x = self.tsm(x)#; print(f"tsm res: {res.shape}")
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            if self.tsm:
                #print(f"tsm x: {x.shape}")
                x += res
                x = self.activation(x)
            else:
                x = self.activation(x)
        return x

## Final architecture
class SeboNet(torch.nn.Module):
    def __init__(
        self, dimension, input_nc, output_nc, num_downs, ngf=24, norm='batch',
        final_act='none', activation='relu', pad_type='reflect', 
        doubleconv=True, residual_connection=False, 
        pooling='Max', interp='nearest', use_skip_connection=True, batch_size=4,
        predict_middle=False, num_timepoints=3,
    ):
        super(SeboNet, self).__init__()
        # Check dims
        ndims = dimension
        assert ndims in [1,2,3,4], 'ndims should be 1--4. found %d' % ndims

        # Decide on whether to use bias based on normalization type
        use_bias = norm == 'instance'
        self.use_bias = use_bias
        self.predict_middle = predict_middle

        
        if ndims == 4:
            Pool = MaxPool4d(kernel_size=2, stride=2)
            ndims = 3
        else:
            Pool = getattr(torch.nn, '%sPool%dd' % (pooling,ndims))
        Conv = getattr(torch.nn, 'Conv%dd' % ndims)
        padding = 'same'

        # Initialize the normalization, activation, and final activation layers
        self.residual_connection = residual_connection
        self.res_dest = []
        self.res_source = []

        # Initial convolution
        encoder = [ConvBlock(ndims=ndims, input_dim=input_nc,
                output_dim=ngf, kernel_size=3, norm='batch', stride=1, padding=1, activation='relu', use_bias=use_bias)]
        
        self.use_skip_connection = use_skip_connection
        self.encoder_idx = []
        in_ngf = ngf

        for i in range(num_downs):
            if i == 0:
                mult = 1
            else:
                mult = 2
            encoder += [ConvBlock(ndims, in_ngf, in_ngf * mult, 3, 1,
                                use_bias, padding=1, norm='batch', tsm=False)]
            encoder += [ConvBlock(ndims, in_ngf * mult, in_ngf * mult, 3, 1,
                                use_bias, padding=1, norm='batch', tsm=True)]
            self.encoder_idx += [len(encoder) - 1]
            if dimension == 4:
                encoder += [MaxPool4d(2,2)]
            else:
                encoder += [Pool(2)]
            in_ngf = in_ngf * mult

        middle = [ConvBlock(ndims, in_ngf, in_ngf * 2, 3, 1, use_bias, padding=1, norm='batch')]
        middle += [ConvBlock(ndims, in_ngf * 2, in_ngf * 2, 3, 1, use_bias, padding=1, tsm=True, norm='batch')]
        
        self.decoder_idx = []
        mult = 2 ** (num_downs)
        for i in range(num_downs):   
            if self.use_skip_connection:
                m = mult + mult // 2
            else:
                m = mult

            if i == 0:
                self.decoder_idx += [len(encoder) + len(middle)]
                decoder = [Upsampling6D(2)]
            else:
                self.decoder_idx += [len(encoder) + len(middle) + len(decoder)]
                decoder += [Upsampling6D(2)]
            decoder += [ConvBlock(ndims, ngf * m, ngf * (mult // 2), 3, 1, use_bias, padding=1, norm='batch')]
            decoder += [ConvBlock(ndims, ngf * (mult // 2),
                        ngf * (mult // 2), 3, 1, use_bias, padding=1, tsm=True, norm='batch')]

            mult    = mult // 2

        if self.predict_middle:
            decoder += [Conv(ngf * mult * num_timepoints, output_nc,
                       3, 1, bias=use_bias, padding=1)]
        else:
            decoder += [Conv(ngf * mult, output_nc,
                        3, 1, bias=use_bias, padding=1)]

        self.encoder = torch.nn.Sequential(*encoder)
        self.middle  = torch.nn.Sequential(*middle)
        self.decoder = torch.nn.Sequential(*decoder)

        self.bs      = batch_size
    
    def final_collapse(self, x):
        x = unravel_time(x, batch_size=self.bs)
        return einops.rearrange(x, 'B T C H W D -> B (T C) H W D')
    
    def forward(self, x):
        x = einops.rearrange(x, 'B C T H W D -> B T C H W D')

        enc_feats = []
        if len(x.shape) == 6:
            x = collapse_time(x)

        for edx, layer in enumerate(self.encoder):
            x = layer(x)
            if self.use_skip_connection and edx in self.encoder_idx:
                enc_feats.append(x)


        for mdx, layer in enumerate(self.middle):
            x = layer(x)


        for ddx, layer in enumerate(self.decoder):
            if ddx == len(self.decoder) - 1 and self.predict_middle:  # Check if it's the last layer
                x = self.final_collapse(x)
            x = layer(x)

            if self.use_skip_connection and (edx + ddx + mdx + 2) in self.decoder_idx:
                x = torch.cat((enc_feats.pop(), x), dim=1)

        x = einops.rearrange(x, '(B C) T H W D -> B T C H W D', B=self.bs)
        return x[:, :, 0, ...] if self.predict_middle else x


# Example case
if __name__ == "__main__":


    net = SeboNet(4, 1, 15, 4, 16, batch_size=4, predict_middle=False, num_timepoints=3).cuda()
    x   = torch.randn(4, 1, 3, 64, 64, 64).cuda()
    y   = net(x)
    print(y.shape)
    from torchinfo import summary
    summary(net, input_data=x)



    """# Instantiate model
    model = TSM_UNet(dimension=3, input_nc=1, output_nc=15,
                 num_downs=4, ngf=24).cuda()
    
    # Get random tensor
    x     = torch.randn(4, 3, 1, 64, 64, 64).cuda() # (B, T, C, H, W, D)

    # Run through the model"
    """
    