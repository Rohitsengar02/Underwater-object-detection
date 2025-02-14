from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import requests
from tqdm import tqdm

class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        
        # Define classes for underwater objects
        self.classes = {
            'fish': 1,
            'coral': 2,
            'starfish': 3,
            'jellyfish': 4,
            'turtle': 5,
            'shark': 6,
            'dolphin': 7,
            'seahorse': 8,
            'crab': 9,
            'octopus': 10
        }
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    # YOLO format to pixel coordinates
                    x_center, y_center, w, h = map(float, data[1:])
                    H, W = image.shape[:2]
                    x1 = int((x_center - w/2) * W)
                    y1 = int((y_center - h/2) * H)
                    x2 = int((x_center + w/2) * W)
                    y2 = int((y_center + h/2) * H)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        
        return image, target

def download_underwater_dataset():
    """Download underwater images and create dataset"""
    dataset_dir = 'datasets/underwater_objects'
    os.makedirs(os.path.join(dataset_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels'), exist_ok=True)
    
    # Define underwater object search terms
    search_terms = [
        'underwater fish', 'coral reef fish', 'tropical fish', 
        'jellyfish underwater', 'sea turtle underwater',
        'shark underwater', 'dolphin underwater', 'seahorse underwater',
        'octopus underwater', 'underwater crab'
    ]
    
    # Download images for each term
    for term in search_terms:
        print(f"Downloading images for: {term}")
        # Use a proper API key and image search API here
        # For example: Bing Image Search API, Google Custom Search API, etc.
        
        # Placeholder for image download code
        # You would implement actual API calls here

def create_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomSizedBBoxSafeCrop(width=640, height=640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Blur(p=0.1),
            A.CLAHE(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(640, 640),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def train_model(num_epochs=50, batch_size=4):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    train_dataset = UnderwaterDataset(
        root_dir='datasets/underwater_objects/train',
        transform=create_transforms(train=True)
    )
    
    val_dataset = UnderwaterDataset(
        root_dir='datasets/underwater_objects/val',
        transform=create_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    
    # Replace the classifier with a new one for our number of classes
    num_classes = len(train_dataset.classes) + 1  # +1 for background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Move model to device
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        # Update learning rate
        lr_scheduler.step()
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation loss: {val_loss/len(val_loader):.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_underwater_model.pth')
            print("Saved best model!")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'models/last_checkpoint.pth')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('datasets/underwater_objects/train/images', exist_ok=True)
    os.makedirs('datasets/underwater_objects/train/labels', exist_ok=True)
    os.makedirs('datasets/underwater_objects/val/images', exist_ok=True)
    os.makedirs('datasets/underwater_objects/val/labels', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Download dataset
    download_underwater_dataset()
    
    # Train model
    train_model()
