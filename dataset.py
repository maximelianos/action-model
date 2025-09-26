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
    def __init__(self, dataset_path, history_length=1, is_validation=False, val_split=0.2, seed=42):
        # Configure your dataset path here
        self.f = h5py.File(dataset_path, "r", swmr=True)
        self.is_validation = is_validation
        self.val_split = val_split
        self.seed = seed
        
        # Get all episodes
        all_episodes = list(self.f["data"].keys())
        debug("robomimic_init", f"Total episodes: {len(all_episodes)}")
        
        # Split episodes by episode number for deterministic train/val split
        np.random.seed(seed)
        episode_indices = np.arange(len(all_episodes))
        np.random.shuffle(episode_indices)
        
        # Calculate split point
        val_episodes_count = int(len(all_episodes) * val_split)
        
        if is_validation:
            # Use last val_episodes_count episodes for validation
            selected_indices = episode_indices[-val_episodes_count:] if val_episodes_count > 0 else []
            self.episodes = [all_episodes[i] for i in selected_indices]
        else:
            # Use remaining episodes for training
            train_episodes_count = len(all_episodes) - val_episodes_count
            selected_indices = episode_indices[:train_episodes_count]
            self.episodes = [all_episodes[i] for i in selected_indices]
        
        debug("dataset_split", f"{'Validation' if is_validation else 'Training'} episodes: {len(self.episodes)}")
        
        # Build episode-to-frames mapping and calculate total length
        self.episode_frames = {}
        self.episode_start_indices = {}
        self.frame_to_episode = []  # Maps global frame index to (episode_name, frame_idx)
        
        total_length = 0
        for episode_name in self.episodes:
            episode_group = self.f[f"data/{episode_name}"]
            num_frames = len(episode_group["obs/image"])
            
            self.episode_frames[episode_name] = num_frames
            self.episode_start_indices[episode_name] = total_length
            
            # Add mapping for each frame in this episode
            for frame_idx in range(num_frames):
                self.frame_to_episode.append((episode_name, frame_idx))
            
            total_length += num_frames
        
        self.length = total_length
        debug("dataset_length", f"{'Validation' if is_validation else 'Training'} dataset length: {self.length}")
        
        # how many frames to input into neural network
        self.history_length = history_length
        

    def load_episode_by_name(self, episode_name: str):
        """
        Load episode by name
        
        Arguments:
            episode_name: str

        Returns:
            episode_group: h5py
        """
        debug("load_episode_name", f"Loading episode: {episode_name}")
        episode_group = self.f[f"data/{episode_name}"]
        debug("load_episode_keys", f"Episode keys: {list(episode_group.keys())}")
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
    
    def preprocess(self, image, action):
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

        # Use different transforms for training vs validation
        if self.is_validation:
            transform = transforms.Compose([
                transforms.ToTensor(), # from numpy to torch tensor
                transforms.Resize((224, 224)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(), # from numpy to torch tensor
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        image = transform(image).float()
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
        # Get the episode and frame index for this global index
        episode_name, frame_idx = self.frame_to_episode[idx]
        episode = self.load_episode_by_name(episode_name)
        
        images = []
        for i in range(self.history_length):
            # Use frame_idx + i for history, but ensure we stay within episode bounds
            history_frame_idx = max(0, frame_idx - self.history_length + i + 1)
            image, action = self.load_frame(episode, history_frame_idx)
            image, action = self.preprocess(image, action)
            images.append(image)
        
        images = torch.cat(images)
        
        # Get the current frame's action
        image, action = self.load_frame(episode, frame_idx)
        image, action = self.preprocess(image, action)
        
        return (
            images,
            action
        )
        
    def __len__(self):
        return self.length
