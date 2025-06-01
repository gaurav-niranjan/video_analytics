import os
import random
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TSNrgbDataset(Dataset):
    def __init__(self, list_file, rgb_root_dir, frame_path, class_map, mode='train', num_segments=4, transform=None):
        
        self.rgb_root_dir = Path(rgb_root_dir)
        self.frame_path = Path(frame_path)
        self.mode= mode
        self.num_segments = num_segments
        self.transform = transform
        self.class_map = class_map

        with open(str(list_file), 'r') as f:
            lines = f.read().splitlines()

        self.samples = []

        for line in lines:
            class_name, video_name = line.strip().split('/')
            label = self.class_map[class_name]
            video_path = self.rgb_root_dir / class_name / video_name
            frame_path = self.frame_path / class_name / video_name
            self.samples.append((video_path, frame_path, label))

    def __len__(self):
        return len(self.samples)
    
    def load_frame(self, video_path, idx):
        frame_path = video_path / f'frame_{idx:05d}.jpg'
        image = Image.open(frame_path).convert("RGB")
        return self.transform(image) if self.transform else image
    
    def __getitem__(self, idx):
        video_path, frame_path, label = self.samples[idx]
        frame_files = sorted(frame_path.glob("frame_*.jpg"))
        num_frames = len(frame_files)

        if num_frames < self.num_segments:
            raise ValueError(f"Video {frame_path} has less frames than segments. Frames = {num_frames}")

        segment_length = num_frames // self.num_segments

        frames = []

        for i in range(self.num_segments):
            start = i * segment_length
            end = (i+1) * segment_length

            if self.mode == "train":
                choosen = random.randint(start, end-1) #end is not included in the current segment
            else:
                choosen = (start+end) // 2

            frame = self.load_frame(frame_path, choosen)
            frames.append(frame)

        video_tensor = torch.stack(frames)

        return video_tensor, label
    
class TSN_Op_flow_Dataset(Dataset):
    def __init__(self, list_file, rgb_root_dir, flow_path, class_map, mode='train', num_segments=4, stack_size=5,transform=None):
        
        self.rgb_root_dir = Path(rgb_root_dir)
        self.flow_path = Path(flow_path)
        self.mode= mode
        self.num_segments = num_segments
        self.transform = transform
        self.class_map = class_map
        self.stack_size = stack_size  # 5 frames x, 5 frames y

        with open(str(list_file), 'r') as f:
            lines = f.read().splitlines()

        self.samples = []

        for line in lines:
            class_name, video_name = line.strip().split('/')
            label = self.class_map[class_name]
            video_path = self.rgb_root_dir / class_name / video_name
            flow_path = self.flow_path / class_name / video_name
            self.samples.append((video_path, flow_path, label))

    def __len__(self):
        return len(self.samples)
    
    def load_flow_stack(self, flow_path, start_idx):
        stack = []

        for i in range(start_idx, start_idx + self.stack_size):
            fx = Image.open(flow_path / f'flow_x_{i:04d}.jpg').convert('L')
            fy = Image.open(flow_path / f"flow_y_{i:04d}.jpg").convert('L')

            if self.transform:
                fx = self.transform(fx)
                fy = self.transform(fy)

            stack.append(fx)
            stack.append(fy)

        return torch.stack(stack, dim=0).squeeze(1) #[10, H, W]
    
    def __getitem__(self, idx):
        video_path, flow_path, label = self.samples[idx]

        flow_files = list(flow_path.glob("flow_x_*.jpg"))
        flow_files.sort()
        num_frames = len(flow_files)

        max_valid_start = num_frames - self.stack_size + 1

        if max_valid_start <  self.num_segments:
            raise ValueError(f"Not enough frames in {flow_path}, has only {num_frames} flow-x frames.")
        
        segment_length = max_valid_start // self.num_segments
        frames = []

        for i in range(self.num_segments):
            start = i * segment_length
            end = (i+1) * segment_length

            if self.mode == 'train':
                idx = random.randint(start, end-1)
            else:
                idx = (start+end)//2

            frame = self.load_flow_stack(flow_path, idx+1) # flow frame filenames start at 1
            frames.append(frame)

        video_tensor = torch.stack(frames)
        return video_tensor, label

