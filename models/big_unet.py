import torch
import re

# Bigger, original model
class ConvBlock(torch.nn.Module):

    def __init__(self, in_features, mid_features, out_features, use_bias, norm_layer='bn'):
        super().__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv3d(in_channels=in_features, out_channels=mid_features, kernel_size=3, padding=1, bias=use_bias)
        self.conv2 = torch.nn.Conv3d(in_channels=mid_features, out_channels=out_features, kernel_size=3, padding=1, bias=use_bias)

        # Batch normalization layers
        if norm_layer  == 'bn':
            self.bn1 = torch.nn.BatchNorm3d(mid_features)
            self.bn2 = torch.nn.BatchNorm3d(out_features)
        elif norm_layer == 'in':
            self.bn1 = torch.nn.InstanceNorm3d(mid_features)
            self.bn2 = torch.nn.InstanceNorm3d(out_features)
        else:
            nf = re.search(r'\d+', norm_layer)
            num_features = int(nf.group())
            self.bn1 = torch.nn.GroupNorm(num_features, mid_features)
            self.bn2 = torch.nn.GroupNorm(num_features, out_features)
            
        

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize the convolutional kernels
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        
        # Initialize the convolutional biases
        if self.conv1.bias is not None:
            torch.nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            torch.nn.init.zeros_(self.conv2.bias)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        return x

class UpConv(torch.nn.Module):

    def __init__(self, in_features, out_features, use_bias):
        super().__init__()
        # Deconvolutional layer
        self.deconv = torch.nn.ConvTranspose3d(in_features, out_features, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1, bias=use_bias, dilation=1, padding_mode='zeros')
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize deconv with Xavier Normal initialization
        torch.nn.init.xavier_normal_(self.deconv.weight)

        # Optionally, initialize the bias to 0
        if self.deconv.bias is not None:
            torch.nn.init.zeros_(self.deconv.bias)


    def forward(self, x, y):
        x = self.deconv(x)
        x = torch.cat((y, x), 1)
        
        return x

class UNet3D(torch.nn.Module):

    def __init__(self, in_features, n_features=64, out_features=15, n_pool=3, use_bias=False, norm_layer='bn'):
        super().__init__()
        print(f'layer norm type: {norm_layer}')
        c_in = in_features
        c_out = n_features
        enc_conv = []
        for _ in range(n_pool):
            enc_conv.append(ConvBlock(c_in, c_out, c_out, use_bias, norm_layer)) # was c_in, c_out//2, c_out
            c_in, c_out = c_out, 2*c_out
        self.enc_conv = torch.nn.ModuleList(enc_conv)

        self.buttom_conv = ConvBlock(c_in, c_out, c_out, use_bias, norm_layer) # was c_in, c_out//2, c_out

        dec_deconv = []
        dec_conv = []
        for _ in range(n_pool):
            dec_deconv.append(UpConv(c_out, c_in, use_bias))
            dec_conv.append(ConvBlock(c_in+c_in, c_in, c_in, use_bias, norm_layer))
            c_out, c_in = c_in, c_in//2
        self.dec_deconv = torch.nn.ModuleList(dec_deconv)
        self.dec_conv = torch.nn.ModuleList(dec_conv)

        # Output layer
        self.out_conv = torch.nn.Conv3d(in_channels=c_out, out_channels=out_features, kernel_size=1, bias=use_bias)
        
        # initialize the weights of the final convolutional layer to be zero's
        torch.nn.init.xavier_normal_(self.out_conv.weight)
        
        
    def forward(self, x):
        feats = []
        
        # Encoder
        for conv in self.enc_conv:
            x = conv(x)
            feats.append(x)
            x = torch.nn.functional.max_pool3d(x, 2, 2)
            
        # Bottom layer
        x = self.buttom_conv(x)
        
        # Decoder
        for deconv, conv, feat in zip(self.dec_deconv, self.dec_conv, feats[::-1]):
            x = deconv(x, feat)
            x = conv(x)
        
        # Output layer
        x = self.out_conv(x)
        return x

if __name__ == "__main__":
    # Generate random tensor
    x = torch.randn(5, 1, 64, 64, 64)

    # Instantiate model
    model = UNet3D(in_features=1, n_features=64, out_features=15, n_pool=3)

    # Run through the model
    print(f"Input: {x.shape} --> Output: {model(x).shape}")