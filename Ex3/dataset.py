import os
import random
from glob import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.nn.utils.rnn import pad_sequence



def collate_fn(batch):
    features = [x[0].T for x in batch]  # (T, D) - > Transposed the feature vector
    labels   = [x[1] for x in batch]    # (T,)

    # Pad along the temporal dimension to the longest T in the batch
    padded_features = pad_sequence(features, batch_first=True)   # (B, T_max, D)
    padded_labels   = pad_sequence(labels, batch_first=True, padding_value=-100)  # (B, T_max)

    return padded_features, padded_labels


class FeatureVecDataset(Dataset):
    def __init__(self, list_file, features_dir, groundTruth_dir, class_map):
        
        self.features_dir = Path(features_dir)
        self.groundTruth_dir = Path(groundTruth_dir)
        self.list_file = Path(list_file) #TRAIN OR TEST.BUNDLE
        self.class_map = class_map

        with open(str(list_file), 'r') as f:
            lines = f.read().splitlines()

        self.samples = []

        for line in lines:
            video_name = line.strip().split('.txt')[0]
            self.samples.append(video_name)

        

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_name = self.samples[idx]
        video_features = np.load(str(self.features_dir / f'{video_name}.npy'))
        video_features = torch.from_numpy(video_features).float()

        with open(str(self.groundTruth_dir / f"{video_name}.txt"), 'r') as f: #Get labels for the current video
            labels = f.read().splitlines()
        labels = [self.class_map[x] for x in labels]  #Change them to integer labels

        labels = torch.tensor(labels, dtype=torch.long)

        if video_features.shape[1] != labels.shape[0]:
            raise ValueError("Video {video_name} does not have complete labels.\n Video shape: {video_features.shape}, labels shape: {}labels.shape")

        return video_features, labels


