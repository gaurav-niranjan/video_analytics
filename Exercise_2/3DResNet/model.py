import torch
import torch.nn as nn

class Block3D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None):


        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1,2,2), padding=(3,3,3), bias = False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2,2,2), padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 and self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):  # [B, 3, T, H, W]
        x = self.relu(self.bn1(self.conv1(x)))  # → [B, 64, T, H/2, W/2]
        x = self.maxpool(x)                     # → [B, 64, T/2, H/4, W/4]

        x = self.layer1(x)  # [B, 64, T/2, H/4, W/4]
        x = self.layer2(x)  # [B, 128, T/4, H/8, W/8]
        x = self.layer3(x)  # [B, 256, T/8, H/16, W/16]
        x = self.layer4(x)  # [B, 512, T/16, H/32, W/32]

        x = self.avgpool(x)  # [B, 512, 1, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

