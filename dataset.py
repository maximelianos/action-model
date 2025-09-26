import os
import h5py
import numpy as np
import json
from pathlib import Path

import torch
import torch.utils.data as data  # dataset shuffling, sample batching, ...
from torchvision import transforms

DEBUG = True
debug_points = {}

def debug(point: str, message: str):
    if DEBUG:
        if not point in debug_points:
            debug_points[point] = True
            print(f"{point}: {message}")


def imginfo(img):
    print(f"shape={img.shape} dtype={img.dtype} min={img.min():.4f} max={img.max():.4f}")


class RobomimicLoader(data.Dataset):
    def __init__(self, dataset_path, history_length=1):
        # Configure your dataset path here
        self.f = h5py.File(dataset_path, "r", swmr=True)
        
        # calculate dataset length
        length = 0
        episodes = list(self.f["data"].keys())
        debug("robomimic_init", f"len(episodes)={len(episodes)}")
        for episode_name in episodes:
            episodes_group = self.f[f"data/{episode_name}"]
            length += len(episodes_group["obs/image"])
        self.length = length
        
        debug("root keys", list(self.f.keys()))
        
        # how many frames to input into neural network
        self.history_length = history_length
        

    def load_episode(self, idx: int):
        """
        Arguments:
            idx: int 

        Returns:
            episode_group: h5py
        """
        episodes = list(self.f["data"].keys())
        debug("load_episode", f"len(episodes) = {len(episodes)}")
        
        episode_name = episodes[idx % len(episodes)]
        debug("load_episode_name", f"Analyzing episode: {episode_name}")

        episode_group = self.f[f"data/{episode_name}"]
        debug("load_episode_keys", f"Episode keys: {list(episode_group.keys())}")
        
        debug("load_episode_type", type(episode_group))
        return episode_group
    
    def load_frame(self, episode_group, idx: int):
        """
        image: (H, W, C)
        action: (7)
        """
        n = len(episode_group["actions"])
        return (
            episode_group["obs/image"][idx % n],
            episode_group["actions"][idx % n]
        )
    
    @staticmethod
    def preprocess(image, action):
        """
        image: np.ndarray (H, W, C)
        action: np.ndarray (7)
        """
        
        # fix image shape - must be (h, w, RGB)
        if len(image.shape) == 2: # (h, w)
            image = image[:, :, np.newaxis]
        if image.shape[2] == 1:
            image = np.tile(image, 3)
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        train_transform = transforms.Compose([
            transforms.ToTensor(), # from numpy to torch tensor
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        val_transform = transforms.Compose([
            transforms.ToTensor(), # from numpy to torch tensor
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = train_transform(image).float()
        action = torch.tensor(action).float()
        
        return (
            image,
            action
        )
    
    def __getitem__(self, idx: int):
        """
        Arguments:
            idx: int 

        Returns:
            image: (C*history, H, W)
            action: (7)
        """
        episode = self.load_episode(idx)
        
        images = []
        for i in range(self.history_length):
            #j = max(0, idx - self.history_length + i + 1)
            j = idx + i
            image, action = self.load_frame(episode, j)
            image, action = self.preprocess(image, action)
            images.append(image)
        
        images = torch.cat(images)
        
        image, action = self.load_frame(episode, idx)
        image, action = self.preprocess(image, action)
        
        return (
            images,
            action
        )
        
    def __len__(self):
        return self.length
