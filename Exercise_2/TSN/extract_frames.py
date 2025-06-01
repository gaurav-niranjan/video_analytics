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

DATA_PATH = Path('/home/gaurav.niranjan/Ex2/data')


CLASS_FILE_PATH = DATA_PATH / 'classes.txt'
FRAME_OUT_DIR = DATA_PATH / 'frames/mini_UCF/'
FLOW_OUT_DIR = DATA_PATH / 'mini_UCF_flow'


def extract_ALL_frames():
    class_folder_paths = []
    dataset_path = DATA_PATH / 'mini_UCF/'
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            class_folder_paths.append(folder)
    for class_fodler in class_folder_paths:
        for video in class_fodler.glob("*.avi"):
            video_name = video.stem
            extract_frames(video, FRAME_OUT_DIR / class_fodler.name / video_name)

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = output_dir / f"frame_{count:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        count += 1

    cap.release()
    print(f"Extracted {count} frames to {output_dir}")

def main():
    extract_ALL_frames()



if __name__ == "__main__":
    main()