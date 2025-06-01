import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import RGB3DResnetDataset
from model import ResNet3D, Block3D
from torchvision.models import resnet18 as resnet2d
from inflation import inflate_resnet
from pathlib import Path



#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#device = torch.device('cuda')

DATA_PATH = Path('/home/gaurav.niranjan/Ex2/data')

CLASS_FILE_PATH = DATA_PATH / 'classes.txt'
FRAME_OUT_DIR = DATA_PATH / 'frames/mini_UCF/'

def build_class_map(classes_file_path):
    with open(str(classes_file_path), 'r') as f:
        lines = f.read().splitlines()
    return {name.strip(): int(idx) for idx, name in (line.split() for line in lines)}

def resnet3d18(num_classes=25):
    return ResNet3D(Block3D, [2, 2, 2, 2], num_classes=num_classes)



def train_epoch(model, dataloader, criterion, optimizer, device, writer=None, global_step=0):
    model.to(device)
    model.train()
    total_loss, total_correct = 0, 0

    for batch_idx, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * labels.size(0)
        total_correct += correct

        if writer:
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Batch_Accuracy", correct / labels.size(0), global_step)

        global_step += 1

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)

    return avg_loss, avg_acc, global_step

def evaluate_multiview(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for clips, labels in dataloader:  # clips: [B, V, 3, T, H, W]
            B, V, C, T, H, W = clips.shape

            clips = clips.view(B * V, C, T, H, W).to(device)  # [B*V, 3, T, H, W]
            labels = labels.to(device)

            outputs = model(clips)                 # [B*V, num_classes]
            outputs = outputs.view(B, V, -1)       # [B, V, num_classes]

            avg_logits = outputs.mean(dim=1)       # [B, num_classes]
            preds = avg_logits.argmax(dim=1)       # [B]

            correct += (preds == labels).sum().item()
            total += B

    acc = correct / total
    print(f"Multi-view Test Accuracy: {acc:.4f}")
    return acc



def main():


    train_list = DATA_PATH / 'train.txt'
    val_list = DATA_PATH / 'validation.txt'

    class_map = build_class_map(CLASS_FILE_PATH)

    num_classes = len(class_map)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--inflate", action="store_true", help="Use 2D ResNet inflation")
    parser.add_argument("--save_path", type=str, default="checkpoint.pth")
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.save_path)

    train_device = torch.device("cuda:0")
    test_device = torch.device("cuda:1")

    # Model
    model = resnet3d18(num_classes=num_classes).to(train_device)
    if args.inflate:
        print("Inflating from 2D ResNet...")
        model2d = resnet2d(pretrained=True)
        inflate_resnet(model, model2d)

    # Datasets
    from torchvision import transforms

    rgb_transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),     # spatial augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    rgb_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),     # deterministic
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RGB3DResnetDataset(train_list, DATA_PATH / 'mini_UCF', FRAME_OUT_DIR, class_map,
                                       num_frames=args.num_frames, mode="train", transform=rgb_transform_train)
    val_dataset = RGB3DResnetDataset(val_list, DATA_PATH / 'mini_UCF', FRAME_OUT_DIR, class_map,
                                     num_frames=args.num_frames, mode="val", transform=rgb_transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")


        train_loss, train_acc, global_step = train_epoch(model, train_loader, criterion, optimizer,
                                                         train_device, writer, global_step)
        val_acc = evaluate_multiview(model, val_loader, test_device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        writer.add_scalar("Train/Epoch_Loss", train_loss, epoch)
        writer.add_scalar("Train/Epoch_Accuracy", train_acc, epoch)

        writer.add_scalar("Validation/Val_Accuracy", val_acc, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{args.save_path}/model.pth')
            print(f"Saved checkpoint to {args.save_path}.")

    # Define hyperparameters
    hparam_dict = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'num_frames': args.num_frames,
        'inflate': args.inflate,
        'epochs': args.epochs
    }

    # Define final metrics
    metric_dict = {
        'best_val_acc': best_val_acc,
    }

    writer.add_hparams(hparam_dict, metric_dict)

    writer.close()


if __name__ == "__main__":
    main()