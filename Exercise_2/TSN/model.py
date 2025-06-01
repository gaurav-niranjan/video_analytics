import torch
import torch.nn as nn
import torchvision.models as models

class TSN_rgb(nn.Module):
    def __init__(self, num_classes, num_segments, pretrained=True):
        super(TSN_rgb, self).__init__()
        self.num_segments = num_segments
        self.num_classes = num_classes

        base_model = models.resnet18(pretrained=pretrained)

        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) #Remove the final layer

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        features = self.feature_extractor(x)
        features = features.view(B, T, 512)

        aggregated = features.mean(dim=1)
        out = self.fc(aggregated)

        return out
    
class TSN_Op_flow(nn.Module):
    def __init__(self, num_classes, num_segments, pretrained=True, first_layer_init=True):
        super(TSN_Op_flow, self).__init__()
        self.num_segments = num_segments
        self.num_classes = num_classes

        base_model = models.resnet18(pretrained=pretrained)

        new_conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=7, padding=2, bias=False)

        if first_layer_init:
            with torch.no_grad():
                old_weights = base_model.conv1.weight
                new_conv1.weight[:] = old_weights.mean(dim=1, keepdim=True).repeat(1, 10, 1, 1)

        base_model.conv1 = new_conv1
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) #Remove the final layer

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        features = self.feature_extractor(x)
        features = features.view(B, T, 512)

        aggregated = features.mean(dim=1)
        out = self.fc(aggregated)

        return out


