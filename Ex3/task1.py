import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from pathlib import Path
import numpy as np
from dataset import FeatureVecDataset, collate_fn
from model import SingleStageTCN
from torch.utils.tensorboard import SummaryWriter

DATA_PATH = Path('./data/')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda')



def build_class_map(classes_file_path):
    with open(str(classes_file_path), 'r') as f:
        lines = f.read().splitlines()
    return {name.strip(): int(idx) for idx, name in (line.split() for line in lines)}



def train_epoch(model, dataloader, criterion, optimizer, writer, global_step, device):
    model.train()
    total_loss, total_correct, total_videos = 0, 0, 0

    for index, (video_feature, labels) in enumerate(dataloader):

        video_feature = video_feature.to(device) #(B, T, D)
        #print(video_feature.shape)
        labels = labels.to(device) #(B,T)

        outputs = model(video_feature) #(B,T,C)

        labels = labels.reshape(-1) #(B*T)
        outputs = outputs.reshape(-1, outputs.shape[-1]) #(B*T, C)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Accuracy
        valid = labels != -100 #To ignore the padding done in the time-steps, since every video could have different time-steps
        preds = outputs.argmax(dim=1)                     # (T,)
        correct = (preds[valid] == labels[valid]).sum().item()
        total_correct += correct
        total_loss += loss.item() * valid.size(0)
        total_videos += valid.sum().item()

    return total_loss / total_videos, total_correct / total_videos

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_videos = 0, 0, 0

    with torch.no_grad():
        for video_features, labels in dataloader:
            video_features = video_features.to(device) #(B, T, D)
            labels = labels.to(device) #(B, T)

            outputs = model(video_features) #(B, T, C)

            labels = labels.reshape(-1) #(B*T)
            outputs = outputs.reshape(-1, outputs.shape[-1])  #(B*T, C)

            loss = criterion(outputs, labels)

            # Accuracy
            valid = labels != -100 #To ignore the padding done in the time-steps, since every video could have different time-steps
            preds = outputs.argmax(dim=1)                     # (T,)
            correct = (preds[valid] == labels[valid]).sum().item()
            total_correct += correct
            total_loss += loss.item() * valid.size(0)
            total_videos += valid.sum().item()

    return total_loss / total_videos, total_correct / total_videos



def main():

    class_map = build_class_map(DATA_PATH / "mapping.txt")

    batch_size = 4
    num_classes = len(class_map)
    input_dim = 2048
    learning_rate = 0.001
    log_dir = './runs/task1'
    writer = SummaryWriter(log_dir=log_dir)
    num_epochs = 50

    train_dataset = FeatureVecDataset(DATA_PATH / "train.bundle", DATA_PATH / "features", DATA_PATH / "groundTruth", class_map)
    val_dataset = FeatureVecDataset(DATA_PATH / "test.bundle", DATA_PATH / "features", DATA_PATH / "groundTruth", class_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    

    model = SingleStageTCN(input_dim=input_dim, num_classes=num_classes, num_layers=10, num_filters = 64).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
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


    torch.save(model.state_dict(), "model.pth")
    writer.close()

if __name__ == "__main__":
    main()
