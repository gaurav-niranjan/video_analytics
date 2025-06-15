import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import random
from torchvision import transforms

class RGB3DResnetDataset(Dataset):
    def __init__(self, list_file, rgb_root_dir, frame_dir, class_map, num_frames, mode='train', transform = None):
        self.frame_dir = frame_dir
        self.mode = mode
        self.num_frames = num_frames
        self.transform = transform
        self.class_map = class_map
        self.rgb_root_dir = rgb_root_dir

        with open(list_file, 'r') as f:
            lines = f.read().splitlines()

        self.samples = []

        for line in lines:
            class_name, video_name = line.strip().split('/')
            label = self.class_map[class_name]
            video_path = self.rgb_root_dir / class_name / video_name
            frame_path = self.frame_dir / class_name / video_name
            self.samples.append((video_path, frame_path, label))

    def _load_frame(self, frame_path):
        img = Image.open(frame_path).convert("RGB")
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, frame_path, label = self.samples[idx]
        frame_files = sorted(list(frame_path.glob("frame_*.jpg")))
        total_frames = len(frame_files)

        if total_frames < self.num_frames:
            raise ValueError(f"Video {video_path} has only {total_frames} frames, requires {self.num_frames}")
        
        if self.mode == 'val' :
            stride = (total_frames - self.num_frames) // 3
            start_indices = [i * stride for i in range(4)]
        else:
            start = random.randint(0, total_frames-self.num_frames)  #Temporal random crop
            start_indices = [start]

        clips = []

        for start in start_indices:
            frames = [self._load_frame(frame_files[start + i]) for i in range(self.num_frames)]

            if self.transform:
                frames = [self.transform(f) for f in frames]
            clip = torch.stack(frames).permute(1,0,2,3) #[3,T, H, W]
            clips.append(clip)

        if self.mode == 'val':
            return torch.stack(clips), label #clips: [4, 3, T, H, W]
        else:
            return clips[0], label
    


