"""
Models for the 3D UNet.

Originally written by Sebo 01/07/2025.

Notes:
    - 'big' model is a standard UNet implementation. From Molin Zhang and Junshen Xu.
    - 'small' model is a more efficient UNet courtesty of Neel Dey and Benjamin Billot
        - used this from the results seen in "Learning General-purpose Biomedical Volume Representations using Randomized Synthesis" Dey et al. (ICLR 2025)

"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from einops import rearrange
import numpy as np


class TemporalShiftUnet(nn.Module):
    """
    U-Net architecture with Temporal Shift Module for spatio-temporal data.
    
    This model extends the standard U-Net by incorporating temporal information
    through TSM modules at different scales of the network.
    
    Parameters
    ----------
    dimension : int
        Number of spatial dimensions (1, 2, or 3).
    input_nc : int
        Number of input channels.
    output_nc : int
        Number of output channels.
    num_downs : int
        Number of downsampling steps.
    n_segment : int
        Number of frames/segments in temporal dimension.
    tsm_mode : str, optional
        Mode for TSM ('shift' or 'residual'), by default 'shift'.
    tsm_levels : list, optional
        Levels of U-Net to apply TSM, by default [0, 1, 2].
        0 is input level, higher numbers are deeper in the network.
    ngf : int, optional
        Number of filters in the last conv layer, by default 24.
    norm : str, optional
        Normalization type, by default 'batch'.
    final_act : str, optional
        Final activation function, by default 'none'.
    activation : str, optional
        Activation function in hidden layers, by default 'relu'.
    pad_type : str, optional
        Padding type, by default 'reflect'.
    doubleconv : bool, optional
        Whether to use double convolution, by default True.
    residual_connection : bool, optional
        Whether to use residual connections, by default False.
    pooling : str, optional
        Pooling type ('Max' or 'Avg'), by default 'Max'.
    interp : str, optional
        Upsampling method, by default 'nearest'.
    use_skip_connection : bool, optional
        Whether to use skip connections, by default True.
    """
    
    def __init__(
        self, dimension, input_nc, output_nc, num_downs, n_segment=3,
        tsm_mode='residual', tsm_levels=[0, 1, 2],
        ngf=24, norm='batch', final_act='none', activation='relu',
        pad_type='reflect', doubleconv=True, residual_connection=False,
        pooling='Max', interp='nearest', use_skip_connection=True, middle_prediction=False,
    ):
        super(TemporalShiftUnet, self).__init__()
        self.unet = Unet(
            dimension, input_nc, output_nc, num_downs, ngf, norm,
            final_act, activation, pad_type, doubleconv, residual_connection,
            pooling, interp, use_skip_connection
        )
        
        self.n_segment = n_segment
        self.tsm_levels = tsm_levels
        self.tsm_mode = tsm_mode
        
        # Create TSM modules for specified levels
        self.tsm_modules = nn.ModuleList()
        
        if 0 in tsm_levels:  # Input level TSM
            self.input_tsm = TemporalShiftModule(
                nn.Identity(), n_segment=n_segment, mode=tsm_mode
            )
        else:
            self.input_tsm = None
            
        # Create TSMs for encoder blocks
        self.encoder_tsms = nn.ModuleList()
        for i in range(num_downs + 1):  # +1 for bottleneck
            if i+1 in tsm_levels:  # +1 because level 0 is reserved for input
                self.encoder_tsms.append(
                    TemporalShiftModule(nn.Identity(), n_segment=n_segment, mode=tsm_mode)
                )
            else:
                self.encoder_tsms.append(None)
                
        # Create TSMs for decoder blocks
        self.decoder_tsms = nn.ModuleList()
        for i in range(num_downs):
            level = num_downs + 2 + i  # +2 because levels 0..num_downs+1 are used by input and encoder
            if level in tsm_levels:
                self.decoder_tsms.append(
                    TemporalShiftModule(nn.Identity(), n_segment=n_segment, mode=tsm_mode)
                )
            else:
                self.decoder_tsms.append(None)
    
    def forward(self, x):
        """
        Forward pass through the Temporal Shift U-Net.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, time, channels, height, width, depth]
            
        Returns
        -------
        torch.Tensor
            Output tensor with same spatial and temporal dimensions
        """
        batch_size, time, channels, height, width, depth = x.shape
        
        # Reshape to [batch_size*time, channels, height, width, depth]
        x = x.reshape(-1, channels, height, width, depth)
        
        # Apply input TSM if needed
        if self.input_tsm is not None:
            x = self.input_tsm(x)
        
        # Forward through UNet with TSM at specific layers
        enc_feats = []
        feat = x
        
        # Encoder path
        encoder_idx = self.unet.encoder_idx
        for layer_id, layer in enumerate(self.unet.model):
            # Apply layer
            feat = layer(feat); print(f"Layer: {layer_id} | Shape: {feat.shape}")
            
            # Apply residual connection if needed
            if self.unet.residual_connection and layer_id in self.unet.res_dest:
                feat_src_idx = self.unet.res_source[self.unet.res_dest.index(layer_id)]
                feat_tmp = self.unet.model[feat_src_idx](feat)
                if feat_tmp.size() == feat.size():
                    feat = feat + 0.1 * feat_tmp
            
            # Check if current layer is the end of an encoder block
            if layer_id in encoder_idx:
                encoder_level = encoder_idx.index(layer_id) + 1  # +1 because level 0 is input
                if encoder_level < len(self.encoder_tsms) and self.encoder_tsms[encoder_level] is not None:
                    # Apply TSM at this encoder level
                    feat = self.encoder_tsms[encoder_level](feat); print(f"E | Layer: {layer_id} | Shape: {feat.shape}")
                
                # Store feature for skip connection
                if self.unet.use_skip_connection:
                    enc_feats.append(feat)
            
            # Check if current layer is the start of a decoder block
            if layer_id in self.unet.decoder_idx:
                decoder_level = self.unet.decoder_idx.index(layer_id)
                
                # Apply skip connection
                if self.unet.use_skip_connection:
                    feat = torch.cat((enc_feats.pop(), feat), dim=1)
                
                # Apply TSM at this decoder level
                if decoder_level < len(self.decoder_tsms) and self.decoder_tsms[decoder_level] is not None:
                    feat = self.decoder_tsms[decoder_level](feat)
        
        # Reshape output back to [batch_size, time, output_channels, height, width, depth]
        output = feat.view(batch_size, time, -1, height, width, depth)
        return output

class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module for enabling temporal information flow in CNNs.
    
    This module shifts part of the channels along the temporal dimension, enabling
    information exchange between neighboring frames without additional parameters.
    
    Parameters
    ----------
    net : nn.Module
        The network to apply temporal shift to.
    n_segment : int
        Number of frames/segments in the temporal dimension.
    n_div : int, optional
        Number of divisions for channel shifting, by default 8.
        A higher value means fewer channels are shifted.
    inplace : bool, optional
        Whether to perform the operation in-place, by default False.
    mode : str, optional
        Shift mode, 'shift' for bidirectional or 'residual' for residual shift, by default 'shift'.
    """
    
    def __init__(self, net, n_segment=3, n_div=3, inplace=False, mode='residual'):
        super(TemporalShiftModule, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.mode = mode
        
        if inplace:
            print('TSM enabled with inplace operation')
        else:
            print('TSM enabled with normal operation')
            
        print(f'TSM using {mode} mode')
            
    def forward(self, x):
        # x shape: [batch_size*n_segment, channels, height, width, depth]
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)
    
    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, mode='residual'):
        """
        Perform the temporal shift operation on input tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size*n_segment, channels, height, width, depth]
        n_segment : int
            Number of frames/segments in the temporal dimension.
        fold_div : int, optional
            Fraction of channels to shift, by default 3.
        inplace : bool, optional
            Whether to perform operation in-place, by default False.
        mode : str, optional
            Shift mode, either 'shift' or 'residual', by default 'shift'.
            
        Returns
        -------
        torch.Tensor
            Tensor after temporal shift operation.
        """
        nt, c, h, w, d = x.size()
        batch_size = nt // n_segment
        x = x.view(batch_size, n_segment, c, h, w, d)
        
        # Default is to use bidirectional shift for half of the channels
        fold = c // fold_div
        
        if mode == 'shift':
            # Shift part of the channels forward and part backward
            if inplace:
                # Shift backward (i.e., current frame uses info from next frame)
                out = x.clone()
                out[:, :-1, :fold] = x[:, 1:, :fold]  # shift backward
                out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift forward
                return out.view(nt, c, h, w, d)
            else:
                # Non-inplace version
                out = torch.zeros_like(x)
                out[:, :-1, :fold] = x[:, 1:, :fold]  # shift backward
                out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]  # shift forward
                out[:, :, 2*fold:] = x[:, :, 2*fold:]  # keep remaining channels unchanged
                return out.view(nt, c, h, w, d)
        elif mode == 'residual':
            # Use residual connection between shifted and original feature maps
            out = torch.zeros_like(x)
            # Shift backward
            out[:, :-1, :fold] = x[:, 1:, :fold] - x[:, :-1, :fold]
            # Shift forward
            out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold] - x[:, 1:, fold:2*fold]
            # Apply residual connection
            out = x + out
            return out.view(nt, c, h, w, d)
        else:
            raise ValueError(f"Unknown TSM mode: {mode}")




import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(ResBlock, self).__init__()
        mid_channels    = out_channels // 2

        self.conv1      = nn.Conv3d(in_channels, mid_channels, kernel_size=1, padding=0)
        self.bn1        = nn.BatchNorm3d(in_channels)
        self.conv2      = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2        = nn.BatchNorm3d(mid_channels)
        self.conv3      = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn3        = nn.BatchNorm3d(out_channels)
        
        if in_channels != out_channels:
            self.conv_skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_skip = None
        self.nl         = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.nl(self.bn1(x))
        if self.conv_skip is not None:
            residual = self.conv_skip(out)
        else:
            residual = out
        
        out = self.conv1(out)
        out = self.nl(self.bn2(out))
        
        out = self.conv2(out)
        out = self.nl(self.bn3(out))
        out = self.conv3(out)
        
        out = out + residual
        return out

class HG(nn.Module):
    def __init__(self, in_channels, depth, nFeat):
        super(HG, self).__init__()
        self.depth = depth
        
        # Encoder pathway
        self.encoders = nn.ModuleList([ResBlock(in_channels, nFeat)])
        for _ in range(depth):
            self.encoders.append(ResBlock(nFeat, nFeat))

        # Decoder pathway
        self.decoders = nn.ModuleList([ResBlock(nFeat, nFeat) for _ in range(depth)])

        # Final output block
        self.final_conv = ResBlock(nFeat, nFeat)

    def forward(self, x):
        encoders = [self.encoders[0](x)]
        
        # Downsampling path
        for i in range(1, self.depth + 1):
            low = F.max_pool3d(encoders[-1], 2, 2)
            low = self.encoders[i](low)
            encoders.append(low)

        # Bottleneck
        decoders = [self.final_conv(encoders[-1])]

        # Skip connections
        skips = [self.encoders[i](enc) for i, enc in enumerate(encoders[:-1])]

        # Upsampling path with skip connections
        for i in range(self.depth):
            low = self.decoders[i](decoders[-1])
            up = F.interpolate(low, scale_factor=2, mode="trilinear", align_corners=False)
            decoders.append(up + skips[-(i + 1)])

        return decoders[-1]

        
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedHG(nn.Module):
    def __init__(self, nFeat, nStacks, depth, nClasses):
        super(StackedHG, self).__init__()
        self.nStacks = nStacks
        self.nFeat = nFeat

        # Head module
        self.head = nn.Sequential(
            nn.Conv3d(1, nFeat // 2, kernel_size=5, padding=2),
            nn.BatchNorm3d(nFeat // 2),
            nn.ReLU(inplace=True),
            ResBlock(nFeat // 2, nFeat),
            ResBlock(nFeat, nFeat),
        )

        # Stacked hourglass modules
        self.hg = nn.ModuleList([HG(nFeat, depth, nFeat).cuda() for _ in range(nStacks)])

        # Post-HG processing layers
        self.res = nn.ModuleList([ResBlock(nFeat, nFeat) for _ in range(nStacks)])
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(nFeat, nFeat, kernel_size=1),
                nn.BatchNorm3d(nFeat),
                nn.ReLU(inplace=True),
            ) for _ in range(nStacks)
        ])
        self.score = nn.ModuleList([nn.Conv3d(nFeat, nClasses, kernel_size=1) for _ in range(nStacks)])

        # Skip connections (for intermediate supervision)
        self.fc_skip = nn.ModuleList([nn.Conv3d(nFeat, nFeat, kernel_size=1) for _ in range(nStacks - 1)])
        self.score_skip = nn.ModuleList([nn.Conv3d(nClasses, nFeat, kernel_size=1) for _ in range(nStacks - 1)])

    def forward(self, x):
        x = self.head(x)
        outputs = []

        for i in range(self.nStacks):
            y = self.hg[i](x)  # Hourglass output
            y = self.res[i](y)  # Residual processing
            y = self.fc[i](y)  # Fully connected layer
            score = self.score[i](y)  # Final heatmap output
            outputs.append(score)

            # Add skip connections if not the last stack
            if i < self.nStacks - 1:
                x = x + self.fc_skip[i](y) + self.score_skip[i](score)

        return outputs  # Returns list of heatmaps from each stack



# HR Bottleneck


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=(
                tensor.dtype
            ),
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc
class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = SinusoidalPositionalEncoding(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels

# VIT pose #TODO
class Embedding(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, opts):
        super().__init__()
        self.image_size     = opts.crop_size
        self.patch_size     = opts.patch_size
        self.num_channels   = 1
        self.hidden_size    = opts.hidden_size
        self.num_patches    = (self.image_size // self.patch_size) ** 3
        self.projection     = nn.Conv3d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.positional_encodings = PositionalEncodingPermute3D(self.hidden_size)
        self.dropout        = nn.Dropout(p=opts.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.projection(x)              # use Convolution to embed the 
        x = x + self.positional_encodings(x)
        x = x.flatten(2).transpose(1, 2)    # flatten to make way for the QKV
        x = self.dropout(x)
        return x

class FFN(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, opts):
        super().__init__()
        self.dense_1 = nn.Linear(opts.hidden_size, opts.intermediate_size)
        self.activation = nn.ReLU()
        self.dense_2 = nn.Linear(opts.intermediate_size, opts.hidden_size)
        self.dropout = nn.Dropout(opts.hidden_dropout_prob)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x



class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.Q = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.K = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.V = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        q,k,v = self.Q(x), self.K(x), self.V(x)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, v)
        return (attention_output, attention_probs)

class MHSA(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.hidden_size = opts.hidden_size
        self.num_attention_heads = opts.num_attention_heads
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = False
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                opts.attention_probs_dropout_prob,
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(opts.hidden_dropout_prob)
    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)

class Block(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.norm1 = nn.LayerNorm(opts.hidden_size)
        self.attn = MHSA(opts)
        self.norm2 = nn.LayerNorm(opts.hidden_size)
        self.ffn = FFN(opts)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

from einops import rearrange
class ClassicDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(config["hidden_size"],config["hidden_size"],2,2),
            nn.BatchNorm3d(config["hidden_size"]),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(config["hidden_size"],config["hidden_size"],2,2),
            nn.BatchNorm3d(config["hidden_size"]),
            nn.ReLU()
        )
        self.output_layer = nn.Conv3d(config["hidden_size"], 15, 1, 1)
        self.image_size = config["image_size"] // config["patch_size"]
    def forward(self, x):
        x = rearrange(x, "B (H W D) C -> B C H W D", H=self.image_size, W=self.image_size, D=self.image_size)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.output_layer(x)
        return x

class SimpleDecoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.image_size = opts.crop_size // opts.patch_size
        self.nl = nn.ReLU()
        self.interpolation = nn.Upsample(scale_factor=opts.patch_size, mode="trilinear")
        self.output_layer = nn.Conv3d(opts.hidden_size,15,3,1,1)
    
    def forward(self, x):
        x = rearrange(x, "B (H W D) C -> B C H W D", H=self.image_size, W=self.image_size, D=self.image_size)
        x = self.nl(x)
        x = self.interpolation(x)
        x = self.output_layer(x)
        return x

class TokenPose(nn.Module):
    def __init__(self, opts, verbose=False):
        super().__init__()
        
        # general setup
        self.opts                   = opts
        self.embedding              = Embedding(self.opts)
        self.encoder                = nn.Sequential(*[Block(self.opts) for _ in range(self.opts.num_layers)] ) # config["num_layers"]
        self.decoder                = ClassicDecoder(self.opts) if self.opts.decoder_type == "classic" else SimpleDecoder(self.opts)
        
        # misc
        self.verbose                = verbose


    def forward(self, x):
        x = self.embedding(x); self.verbose and print(f"After positional embedding {x.shape}")
        x = self.encoder(x); self.verbose and print(f"After encodder {x.shape}")
        x = self.decoder(x); self.verbose and print(f"After decoder {x.shape}")
        return x





def window_partition(x, window_size):
    """
    Partition a 5D tensor (B, C, H, W, D) into non-overlapping windows.
    """
    B, C, H, W, D = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
    # Move window dimensions together and flatten
    windows = rearrange(x, "B C h ws w ws2 d ws3 -> B (h w ws ws2 d ws3) C")
    # Alternatively, you could return windows of shape (num_windows*B, window_size, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, B, C, H, W, D):
    """
    Reverse the window partitioning.
    """
    # Here, we assume windows is of shape (B, num_windows, C) and need to reshape back.
    # This function would need to reverse the rearrange in window_partition.
    # The exact implementation depends on your chosen partitioning scheme.
    # For illustration, we'll assume a simple reverse:
    x = windows.view(B, C, H, W, D)
    return x

class ShiftedWindowAttention(nn.Module):
    def __init__(self, hidden_size, window_size, shift_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size  # e.g., 2 or 4 depending on your design
        self.shift_size = shift_size    # typically window_size // 2
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size

        self.qkv = nn.Linear(hidden_size, self.all_head_size * 3, bias=True)
        self.proj = nn.Linear(self.all_head_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H, W, D):
        # x shape: (B, N, C) where N = H*W*D or arranged as windows later
        B, N, C = x.shape

        # Reshape tokens to 3D feature map
        x = x.transpose(1, 2).view(B, C, H, W, D)

        # Apply cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(2, 3, 4))
        else:
            shifted_x = x

        # Partition windows
        # For simplicity, assume window_partition returns a tensor of shape (num_windows*B, window_size^3, C)
        windows = self._window_partition(shifted_x)  # implement similar to the 2D/3D case
        # Now windows shape: (num_windows*B, window_volume, C)

        # Compute Q, K, V for windows
        qkv = self.qkv(windows)  # shape: (num_windows*B, window_volume, 3*all_head_size)
        qkv = qkv.reshape(windows.shape[0], windows.shape[1], 3, self.num_heads, self.attention_head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each has shape (num_windows*B, num_heads, window_volume, attention_head_size)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.attention_head_size)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (num_windows*B, num_heads, window_volume, attention_head_size)
        out = out.transpose(1, 2).reshape(windows.shape[0], windows.shape[1], self.all_head_size)
        out = self.proj(out)
        out = self.dropout(out)

        # Merge windows back to feature map
        shifted_x = self._window_reverse(out, B, C, H, W, D)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(2, 3, 4))
        else:
            x = shifted_x

        # Flatten back to (B, N, C)
        x = x.view(B, C, -1).transpose(1, 2)
        return x

    def _window_partition(self, x):
        """
        Partition the 3D feature map x (B, C, H, W, D) into windows.
        """
        B, C, H, W, D = x.shape
        window_size = self.window_size
        # Ensure H, W, D are divisible by window_size
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
        windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        windows = windows.view(-1, window_size ** 3, C)  # (num_windows*B, window_volume, C)
        return windows

    def _window_reverse(self, windows, B, C, H, W, D):
        """
        Reverse windows back to the 3D feature map.
        """
        window_size = self.window_size
        windows = windows.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, C)
        windows = windows.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = windows.view(B, C, H, W, D)
        return x

# Example of integrating ShiftedWindowAttention in a Block
class ShiftedWindowBlock(nn.Module):
    def __init__(self, config, window_size, shift_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(config["hidden_size"])
        self.attn = ShiftedWindowAttention(
            hidden_size=config["hidden_size"],
            window_size=window_size,
            shift_size=shift_size,
            num_heads=config["num_attention_heads"],
            dropout=config["hidden_dropout_prob"]
        )
        self.norm2 = nn.LayerNorm(config["hidden_size"])
        self.ffn = FFN(config)
    
    def forward(self, x, H, W, D):
        # x is expected to be (B, N, C) where N = H*W*D
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, H, W, D)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + x
        return x

# In your main model (TokenPose), you could alternate between global MHSA Blocks and shifted window blocks:
class TokenPoseWithShiftedWindow(nn.Module):
    def __init__(self, config, window_size=2):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches, config["hidden_size"]))
        
        # For simplicity, alternate between global attention and shifted window attention blocks
        self.blocks = nn.ModuleList()
        for i in range(config["num_layers"]):
            if i % 2 == 0:
                # Global attention block
                self.blocks.append(Block(config))
            else:
                # Shifted window attention block
                shift_size = window_size // 2  # for example
                self.blocks.append(ShiftedWindowBlock(config, window_size, shift_size))
        
        self.decoder = ClassicDecoder(config) if config["decoder_type"] == "classic" else SimpleDecoder(config)
    
    def forward(self, x):
        x = self.patch_embeddings(x) + self.position_embeddings  # (B, N, C)
        # Assuming the patches form a cubic grid
        num_patches = self.patch_embeddings.num_patches
        side = round(num_patches ** (1/3))  # compute approximate side length of 3D grid
        H = W = D = side
        for block in self.blocks:
            # Check if block accepts extra spatial dims (for shifted window blocks)
            if isinstance(block, ShiftedWindowBlock):
                x = block(x, H, W, D)
            else:
                x = block(x)
        x = self.decoder(x)
        return x



# Helper to test the models
if __name__ == "__main__":
    """
    opts = {'nJoints': 15,
            'nFeat': 64, 
            'depth': 3,

                }
    opts = type('opts', (object,), opts)
    # Test the model
    model   = Unet(
                dimension=3,    # Only 3D supported for now
                input_nc=1,     # number of input channels
                output_nc=15,   # number of output channels
                num_downs=4,    # number of downsampling layers
                ngf=16,         # channel multiplier
                )
    #x       = torch.randn(1, 1, 64, 64, 64) # 
    
    model = DSNT(opts, n_locations=15)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    print(y[0].shape, torch.sum(y[1]))
    from torchinfo import summary
    #summary(model, input_data=x)
    """
    
    # Input sha"pe: (BS, T, C, H, W, D) = (16, 3, 1, 64, 64, 64)
    batch_size, time_steps, channels, height, width, depth = 16, 3, 1, 64, 64, 64
    input_tensor = torch.randn(batch_size, time_steps, channels, height, width, depth)#.cuda()
    
    # Create model
    model1 = TemporalShiftUnet(
        dimension=3,            # 3D data
        input_nc=channels,      # 1 input channel
        output_nc=15,            # 1 output channel
        num_downs=4,            # 4 downsampling steps
        n_segment=time_steps,   # 3 time steps
        tsm_levels=[0, 1],   # Apply TSM at input, first and second level
        ngf=16,                 # Base filter count
        norm='batch',           # Batch normalization
        use_skip_connection=True
    )#.cuda()

    model2 = Unet(
                dimension=3,    # Only 3D supported for now
                input_nc=1,     # number of input channels
                output_nc=15,   # number of output channels
                num_downs=4,    # number of downsampling layers
                ngf=16,         # channel multiplier
                )
    
    # Forward pass
    print(input_tensor.shape)

    output = model1(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
   # print(f"Output shape: {output.shape}")
    print(f"Model1 parameters: {sum(p.numel() for p in model1.parameters())}")
    print(f"Model2 parameters: {sum(p.numel() for p in model2.parameters())}")
    """
    

    x = torch.randn((4, 1, 8, 64, 64, 64)).cuda()
    model = Unet(dimension=4, input_nc=1, output_nc=15, num_downs=3, ngf=32, pad_type="zeros").cuda()
    y = model(x)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(y.shape)
    
    #model = HRNet(in_channels=1, out_channels=64)
    #y = model(x)
    
    
    
    opts = {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "crop_size": 96,
        "num_channels": 1,
        "patch_size": 8,
        "num_attention_heads": 6,
        "qkv_bias": False,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "num_layers": 8,
        "decoder_type": "simple"

    }
    opts = type('opts', (object,), opts)
    from torchinfo import summary
    model = TokenPose(opts).cuda()
    x = torch.randn(1, 1, opts.crop_size, opts.crop_size, opts.crop_size).cuda()
    print(model(x).shape)
    summary(model)
    """