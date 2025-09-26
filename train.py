#!/usr/bin/env python3

import os
from glob import glob
from pathlib import Path
import time
import datetime as dt

import numpy as np
import cv2
from skimage.io import imread

import torch
import torch.utils.data as data
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, get_model
import timm
import torchsummary

import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import RobomimicLoader
from imitation_model import CNN
from loss_logger import LossLogger
from plot_loss import plot_training_curves


def imginfo(img):
    print(type(img), img.dtype, img.shape, img.min(), img.max())


class Vit(nn.Module):
    def __init__(self, input_shape=None, num_classes=17):
        super().__init__()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.head = nn.Sequential(
            nn.Linear(self.vit.head.in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, images):
        # images: (b, c, h, w)
        batch_size, channels, h, w = images.shape
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        images = transform(images) # (c, h, w)
        
        # if input image has 1 channel, repeat the torch tensor
        if channels < 3:
            images = images.repeat(1, 3, 1, 1)

        x = self.vit(images) # (b, c, h, w) -> (b, 768) -> (b, num_classes)
        return x


# Default constants (will be overridden by config)
ACTION_DIM = 7


def lrfn(epoch, args):
    """Learning rate schedule function"""
    rampup_epochs = args.get('rampup_epochs', 0)
    start_lr = args.get('start_lr', 0.0001)
    sustain_epochs = args.get('sustain_epochs', 0)
    min_lr = args.get('min_lr', 0.001)
    max_lr = args.get('max_lr', 0.01)
    exp_decay = args.get('exp_decay', 0.8)
    step_epoch = args.get('step_epoch', [])
    step_lr = args.get('step_lr', [])
    
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        for i in range(len(step_epoch)-1, -1, -1):
            if epoch > step_epoch[i]:
                return step_lr[i]
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr


def create_datasets(args):
    """Create training and validation datasets with episode-based split"""
    print("Loading dataset...")
    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset file doesn't exist: {args.dataset_path}")
    
    # Create training and validation datasets with episode-based split
    train_dataset = RobomimicLoader(
        args.dataset_path, 
        history_length=args.history_length,
        is_validation=False,
        val_split=args.val_split,
        seed=args.seed
    )
    
    val_dataset = RobomimicLoader(
        args.dataset_path,
        history_length=args.history_length, 
        is_validation=True,
        val_split=args.val_split,
        seed=args.seed
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, args, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': args
    }
    
    # Save latest checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / 'latest_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best checkpoint if this is the best model
    if is_best:
        best_path = Path(args.checkpoint_dir) / 'best_checkpoint.pth'
        torch.save(checkpoint, best_path)
        print(f"Best checkpoint saved to {best_path}")


def load_checkpoint(model, optimizer, args):
    """Load model checkpoint if available"""
    checkpoint_path = Path(args.checkpoint_dir) / 'latest_checkpoint.pth'
    
    if checkpoint_path.exists() and args.resume:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("Starting training from scratch")
        return 0


def train_epoch(model, train_loader, criterion, optimizer, epoch, args, logger):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    num_samples = 0
    
    # Update learning rate
    for g in optimizer.param_groups:
        g['lr'] = lrfn(epoch, args)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_samples += inputs.size(0)
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    
    # Log training loss
    logger.append({
        't': epoch,
        'train': avg_loss
    })
    
    return avg_loss


def validate_epoch(model, val_loader, criterion, epoch, args, logger):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            num_samples += inputs.size(0)
    
    avg_loss = running_loss / len(val_loader)
    
    # Log validation loss
    logger.append({
        't': epoch,
        'val': avg_loss
    })
    
    return avg_loss


def train_model(model, train_dataset, val_dataset, args):
    """Main training loop"""
    print("Setting up training...")
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=False
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=False
    )
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup logging
    loss_log_path = Path(args.output_dir) / "loss.h5"
    logger = LossLogger(str(loss_log_path), overwrite=not args.resume)
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if resuming
    start_epoch = load_checkpoint(model, optimizer, args)
    
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Device: {args.device}")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, args, logger)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, epoch, args, logger)
        
        print(f'Epoch {epoch+1}/{args.num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, args, is_best)
        
        # Save loss logger periodically
        if epoch % args.save_freq == 0:
            logger.save()
    
    # Final save
    logger.save()
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    return logger


def create_model(args):
    """Create model based on configuration"""
    print("Creating model...")
    
    if args.model_type == 'cnn':
        model = CNN(input_channels=args.input_channels, n_classes=ACTION_DIM)
    elif args.model_type == 'vit':
        model = Vit(num_classes=ACTION_DIM)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(args.device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def test_model_inference(model, dataset, args):
    """Test model inference with sample data"""
    print("Testing model inference...")
    model.eval()
    
    # Get a few test samples
    test_indices = [0, 3, 10] if len(dataset) > 10 else [0]
    
    with torch.no_grad():
        for idx in test_indices:
            image, action = dataset[idx]
            image = image[None, ...].to(args.device)
            
            output = model(image)[0].detach().cpu()
            
            print(f"\nSample {idx}:")
            print(f"Ground truth: {action}")
            print(f"Model output: {output}")
            
            # Calculate accuracy within threshold
            thr = 0.2
            accuracy = (output - action).abs() < thr
            print(f"Accuracy within {thr}: {accuracy}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args: DictConfig):
    """Main execution function"""
    print("=== Action Model Training ===\n")
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    args.device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {args.device}")
    print(f"Configuration:\n{OmegaConf.to_yaml(args)}")
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args)
    
    # Test dataset sample
    sample_image, sample_action = train_dataset[0]
    
    print(f"\nSample data info:")
    imginfo(sample_image)
    imginfo(sample_action)
    
    # Create model
    model = create_model(args)
    
    # Test model with sample
    sample_image_batch = sample_image[None, ...].to(args.device)
    sample_output = model(sample_image_batch)
    print(f"Model output shape: {sample_output.shape}")
    
    # Train model
    logger = train_model(model, train_dataset, val_dataset, args)
    
    # Plot training curves
    plot_training_curves(logger, args)
    
    # Test final model
    test_model_inference(model, val_dataset, args)
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()