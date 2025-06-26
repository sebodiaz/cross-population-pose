import torch


class HRBottlneck(torch.nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HRBottlneck, self).__init__()
        self.conv1 = torch.nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = torch.nn.BatchNorm3d(planes)
        self.conv2 = torch.nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = torch.nn.BatchNorm3d(planes)
        self.conv3 = torch.nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = torch.nn.BatchNorm3d(planes * self.expansion)
        self.relu  = torch.nn.ReLU()
        self.ds    = downsample
        self.s     = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.ds is not None:
            residual = self.ds(x)
        out += residual
        out = self.relu(out)
        return out


# HRNet #TODO implement this
class HRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inplanes = 64

        # initial stem input layer // first sentence, Section 3 in HR-Net paper
        self.stem = torch.nn.Sequential(
                                torch.nn.Conv3d(in_channels  = in_channels,
                                          out_channels = out_channels,
                                          kernel_size  = 3,
                                          padding      = 1,
                                          stride       = 2, 
                                          bias         = False),
                                torch.nn.BatchNorm3d(out_channels),
                                torch.nn.ReLU(),
                                torch.nn.Conv3d(in_channels  = out_channels,
                                          out_channels = out_channels,
                                          kernel_size  = 3,
                                          padding      = 1,
                                          stride       = 2, 
                                          bias         = False),
                                torch.nn.BatchNorm3d(out_channels),
                                torch.nn.ReLU(),
        )

        # layer 1
        self.layer1 = self._make_layer(HRBottlneck, 64, 4) 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return torch.nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.stem(x); print(x.shape)
        x = self.layer1(x); print(x.shape)
        return x
    

if __name__ == "__main__":
    # Generate random input
    x = torch.randn(5, 1, 64, 64, 64)

    # Instantiate model
    model = HRNet(in_channels=1, out_channels=15)
    
    # Run through the model
    print(f"Input: {x.shape} --> Output: {model(x).shape}")
