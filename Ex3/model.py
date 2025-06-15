import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)

        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return x + out
    
class SingleStageTCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers = 10, num_filters = 64):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, num_filters, kernel_size=1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(num_filters, num_filters, dilation=i+1) for i in range(num_layers)
        ])
        self.output_layer = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        #x: (B, T, D)
        x = x.permute(0,2,1) # (B, D, T) Since conv 1d operates on time dimension
        out = self.input_proj(x)

        for layer in self.layers:
            out = layer(out)

        out = self.output_layer(out)

        return out.permute(0,2,1) #(B, T, C)    C: No. of classes
    
class MultiStageTCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_stages=4, num_layers=10, num_filters=64):
        super().__init__()

        self.num_stages = num_stages
        self.stage1 = SingleStageTCN(input_dim, num_classes, num_layers, num_filters)
        self.stages = nn.ModuleList([
            SingleStageTCN(input_dim+num_classes, num_classes, num_layers, num_filters) for _ in range(num_stages-1)
        ])

    def forward(self, x):
        #x: (B, T, D)
        out = self.stage1(x) #(B, T, C)
        probs = F.softmax(out, dim=-1)
        outputs = [out]

        for stage in self.stages:
            inp = torch.cat([probs, x], dim=2) #(B, T, C+D)
            out = stage(inp)
            probs = F.softmax(out, dim=-1)
            outputs.append(out)

        return outputs #list of (B, T, C)
    
class MultiScaleTCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=10, num_filters=64):
        super().__init__()
        self.num_classes = num_classes

        # One TCN per resolution
        self.tcn_full = SingleStageTCN(input_dim, num_classes, num_layers, num_filters)
        self.tcn_4x = SingleStageTCN(input_dim, num_classes, num_layers, num_filters)
        self.tcn_8x = SingleStageTCN(input_dim, num_classes, num_layers, num_filters)

        # Final 1x1 conv to merge
        self.merge_conv = nn.Conv1d(3 * num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # Full-resolution
        out_full = self.tcn_full(x)          # (B, T, C)

        # 4x Downsampling
        x_4x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=4, stride=4).permute(0, 2, 1)  # (B, T//4, D)
        out_4x = self.tcn_4x(x_4x)           # (B, T//4, C)
        out_4x_up = F.interpolate(out_4x.permute(0, 2, 1), size=T, mode='nearest').permute(0, 2, 1)  # (B, T, C)

        # 8x Downsampling
        x_8x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=8, stride=8).permute(0, 2, 1)  # (B, T//8, D)
        out_8x = self.tcn_8x(x_8x)           # (B, T//8, C)
        out_8x_up = F.interpolate(out_8x.permute(0, 2, 1), size=T, mode='nearest').permute(0, 2, 1)  # (B, T, C)

        # Concatenate
        merged = torch.cat([out_full, out_4x_up, out_8x_up], dim=2)  # (B, T, 3C)
        merged = merged.permute(0, 2, 1)  # (B, 3C, T) for Conv1d

        # Final classification
        out = self.merge_conv(merged)     # (B, C, T)
        return out.permute(0, 2, 1), out_4x_up, out_8x_up #Select the approprite output during training