import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
import cv2
from pathlib import Path
from dataset import TSNrgbDataset
from model import TSN_rgb
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda')

DATA_PATH = Path('/home/gaurav.niranjan/Ex2/data')

CLASS_FILE_PATH = DATA_PATH / 'classes.txt'
FRAME_OUT_DIR = DATA_PATH / 'frames/mini_UCF/'
FLOW_OUT_DIR = DATA_PATH / 'mini_UCF_flow'

def build_class_map(classes_file_path):
    with open(str(classes_file_path), 'r') as f:
        lines = f.read().splitlines()
    return {name.strip(): int(idx) for idx, name in (line.split() for line in lines)}



def train_epoch(model, dataloader, criterion, optimizer, writer, global_step, device):
    model.train()
    total_loss, total_correct = 0, 0

    for index, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Train/Batch_Loss", loss.item() * labels.size(0), global_step + index)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


def main():
    train_list = DATA_PATH / 'train.txt'
    val_list = DATA_PATH / 'validation.txt'

    log_dir = './runs/tsn_rgb/run1/bs_64'
    writer = SummaryWriter(log_dir=log_dir)
    batch_size = 64
    num_epochs = 30
    num_segments = 4
    learning_rate = 1e-3

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    class_map = build_class_map(CLASS_FILE_PATH)

    num_classes = len(class_map)

    train_dataset = TSNrgbDataset(train_list, DATA_PATH / 'mini_UCF', FRAME_OUT_DIR ,class_map, mode='train', transform=transform)
    val_dataset = TSNrgbDataset(val_list, DATA_PATH / 'mini_UCF', FRAME_OUT_DIR, class_map, mode='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TSN_rgb(num_classes=num_classes, num_segments=num_segments, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    global_step = 0

    for epoch in range(num_epochs):

        epoch_loss, epoch_accuracy = train_epoch(model, train_loader, criterion, optimizer, writer, global_step, device)

        writer.add_scalar("Train/Epoch_Loss", epoch_loss, global_step)
        writer.add_scalar("Train/Epoch_Accuracy", epoch_accuracy, global_step)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("Val/Val_Loss", val_loss, global_step)
        writer.add_scalar("Val/Val_Accuracy", val_accuracy, global_step)

        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        global_step+=1


    torch.save(model.state_dict(), "tsn_rgb.pth")
    writer.close()

    pass


if __name__ == "__main__":
    main()
