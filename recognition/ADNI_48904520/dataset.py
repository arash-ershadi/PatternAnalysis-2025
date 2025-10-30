import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image

class ADNIDataset(Dataset):
    """
    Dataset class for ADNI brain MRI classification
    Two classes: AD (Alzheimer's Disease) and NC (Normal Cognitive)
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Base directory containing AD_NC folder
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.root_dir = os.path.join(root_dir, "AD_NC", split)
        self.transform = transform
        self.samples = []

        # Verify directory exists
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Directory does not exist: {self.root_dir}")

        # Load samples for each class
        for label_name, label in [("AD", 1), ("NC", 0)]:
            class_dir = os.path.join(self.root_dir, label_name)

            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue

            # Get all image files
            files = [f for f in os.listdir(class_dir) if f.endswith('.jpeg')]

            for fname in files:
                full_path = os.path.join(class_dir, fname)
                self.samples.append((full_path, label))

        print(f"Loaded {len(self.samples)} samples for {split} split")

        # Print class distribution
        ad_count = sum(1 for _, label in self.samples if label == 1)
        nc_count = sum(1 for _, label in self.samples if label == 0)
        print(f"  AD: {ad_count}, NC: {nc_count}")

    def __getitem__(self, idx):
        # Load image file
        img_path, label = self.samples[idx]

        try:
            # Load image as grayscale
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('L', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

def get_dataloaders(root_dir, batch_size=32, num_workers=4,
                    val_split=0.2):
    """
    Create train, validation, and test dataloaders

    Args:
        root_dir: Base directory containing ADNI data (e.g., /home/groups/comp3710/ADNI)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        val_split: Proportion of training data to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    # Enhanced training transforms with MORE aggressive augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize larger first
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.9, 1.1),
            shear=10
        ),
        # Add color jitter for intensity variations
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # Add Gaussian blur occasionally
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        # Use dataset-specific mean and std
        transforms.Normalize(mean=[0.1159], std=[0.2199])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1159], std=[0.2199])
    ])

    # Load train and test data (data is already split)
    train_dataset = ADNIDataset(root_dir, split='train', transform=train_transform)
    test_dataset = ADNIDataset(root_dir, split='test', transform=val_transform)
    val_dataset = ADNIDataset(root_dir, split="train", transform=val_transform)

    # Split training data into train and validation
    train_size = int((1 - val_split) * len(train_dataset))

    indices = list(range(len(train_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader