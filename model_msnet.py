import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # Reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # Transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # Flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MBlock(nn.Module):
    """
    Structure: PW Conv -> DW Conv -> PW Conv
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(MBlock, self).__init__()
        # Pointwise Conv
        self.pw_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Depthwise Conv
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Pointwise Conv (Project)
        self.pw_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pw_conv1(x)
        x = self.dw_conv(x)
        x = self.pw_conv2(x)
        return x

class SBlock(nn.Module):
    """
    S Block: Channel Split -> Parallel Branch/Pass Through -> Concat -> Shuffle
    """
    def __init__(self, in_channels):
        super(SBlock, self).__init__()
        self.split_channels = in_channels // 2
        
        # Convolution branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, 
                      padding=1, groups=self.split_channels, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Channel Split
        x1, x2 = torch.split(x, self.split_channels, dim=1)
        # Branch processing
        x2 = self.conv_branch(x2)
        # Concat
        out = torch.cat((x1, x2), dim=1)
        # Channel Shuffle
        return channel_shuffle(out, 2)

class MSNet(nn.Module):
    def __init__(self, num_classes=80):
        super(MSNet, self).__init__()
        
        # Initial 3x3 Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 7 M Blocks
        self.m_blocks = nn.Sequential(
            MBlock(32, 64, stride=1),
            MBlock(64, 64, stride=1),
            MBlock(64, 128, stride=2), # 128->64
            MBlock(128, 128, stride=1),
            MBlock(128, 128, stride=1),
            MBlock(128, 256, stride=2), # 64->32
            MBlock(256, 256, stride=1)
        )

        # 2 S Blocks
        self.s_blocks = nn.Sequential(
            SBlock(256),
            SBlock(256)
        )

        # Final 1x1 Conv
        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input 128x128
        x = self.conv1(x)       # -> 64x64
        x = self.m_blocks(x)    # -> 16x16
        x = self.s_blocks(x)    # -> 16x16
        x = self.conv_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model(num_classes=80):
    return MSNet(num_classes=num_classes)