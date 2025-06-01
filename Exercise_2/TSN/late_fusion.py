import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from dataset import TSN_Op_flow_Dataset, TSNrgbDataset
from model import TSN_Op_flow, TSN_rgb
from main_flow import build_class_map

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda')

DATA_PATH = Path('/home/gaurav.niranjan/Ex2/data')

CLASS_FILE_PATH = DATA_PATH / 'classes.txt'
FRAME_OUT_DIR = DATA_PATH / 'frames/mini_UCF/'
FLOW_OUT_DIR = DATA_PATH / 'mini_UCF_flow'

flow_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Converts grayscale to shape (1, H, W)
    transforms.Normalize(mean=[0.5], std=[0.226]) #For flow images
    ])

rgb_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def main():

    class_map = build_class_map(CLASS_FILE_PATH)
    val_list = DATA_PATH / 'validation.txt'

    num_classes = len(class_map)

    model_rgb = TSN_rgb(num_classes=num_classes, num_segments=4)
    model_rgb.load_state_dict(torch.load("/home/gaurav.niranjan/Ex2/TSN/runs/tsn_rgb/run1/bs_64/tsn_rgb.pth"))
    model_rgb.eval().to(device)

    model_flow = TSN_Op_flow(num_classes=num_classes, num_segments=4)
    model_flow.load_state_dict(torch.load("/home/gaurav.niranjan/Ex2/TSN/runs/tsn_OF/run2/random_init/tsn_OF.pth"))
    model_flow.eval().to(device)

    rgb_dataset = TSNrgbDataset(val_list, DATA_PATH / 'mini_UCF', FRAME_OUT_DIR, class_map, mode="val", transform=rgb_transform)
    flow_dataset = TSN_Op_flow_Dataset(val_list, DATA_PATH / 'mini_UCF', FLOW_OUT_DIR, class_map, mode="val", transform=flow_transform)

    rgb_loader = DataLoader(rgb_dataset, batch_size=1, shuffle=False)
    flow_loader = DataLoader(flow_dataset, batch_size=1, shuffle=False)


    correct = 0
    total = 0

    with torch.no_grad():
        for (rgb_data, label_rgb), (flow_data, label_flow) in zip(rgb_loader, flow_loader):
            assert label_rgb.item() == label_flow.item(), "Mismatched labels!"

            rgb_data = rgb_data.to(device)
            flow_data = flow_data.to(device)
            label = label_rgb.to(device)

            # Get predictions (logits)
            logits_rgb = model_rgb(rgb_data)     # shape: [1, num_classes]
            logits_flow = model_flow(flow_data)  # shape: [1, num_classes]

            # Convert to probabilities (optional for late fusion)
            probs_rgb = F.softmax(logits_rgb, dim=1)
            probs_flow = F.softmax(logits_flow, dim=1)

            # Late Fusion (average)
            fused_probs = (probs_rgb + probs_flow) / 2
            pred = fused_probs.argmax(dim=1)

            if pred.item() == label.item():
                correct += 1
            total += 1

    acc = correct / total
    print(f"Late Fusion Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()