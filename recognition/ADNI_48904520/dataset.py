import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image


class ADNIDataset(Dataset):
    """
    Dataset class for ADNI brain MRI classification
    Two classes: AD (Alzheimer's Disease) and NC (Normal Cognitive)
    """

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Base directory containing AD_NC folder
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Load images based on split
        if split == 'train':
            split_dir = os.path.join(data_dir, 'AD_NC', 'train')
        elif split == 'test':
            split_dir = os.path.join(data_dir, 'AD_NC', 'test')
        else:
            # For validation, use train folder and split later
            split_dir = os.path.join(data_dir, 'AD_NC', 'train')

        # Load AD images (label = 1)
        ad_dir = os.path.join(split_dir, 'AD')
        print(ad_dir)
        if os.path.exists(ad_dir):
            print(ad_dir)
            ad_files = glob.glob(os.path.join(ad_dir, '*.jpeg'))
            ad_files.sort()
            self.image_files.extend(ad_files)
            self.labels.extend([1] * len(ad_files))

        # Load NC images (label = 0)
        nc_dir = os.path.join(split_dir, 'NC')
        if os.path.exists(nc_dir):
            nc_files = glob.glob(os.path.join(nc_dir, '*.jpeg'))
            nc_files.sort()
            self.image_files.extend(nc_files)
            self.labels.extend([0] * len(nc_files))

    def __getitem__(self, idx):
        # Load image file
        img_path = self.image_files[idx]

        # Load image as grayscale
        try:
            image = Image.open(img_path).convert('L')
        except:
            # If image loading fails, try with PIL
            image = Image.fromarray(np.random.rand(256, 256) * 255).convert('L')

        # Convert to numpy array
        image = np.array(image).astype(np.float32)

        # Resize to fixed size (224, 224) for ConvNeXt
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)  # Add channel dimension (1, H, W)

        target_size = (224, 224)
        image = F.interpolate(image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        image = image.squeeze(0)

        # Normalize to [0, 1] first
        image = image / 255.0

        # Then normalize with mean and std
        image = (image - image.mean()) / (image.std() + 1e-8)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.image_files)


def get_dataloaders(data_dir, batch_size=8, num_workers=4,
                    val_split=0.15):
    """
    Create train, validation, and test dataloaders

    Args:
        data_dir: Base directory containing ADNI data (e.g., /home/groups/comp3710/ADNI)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        val_split: Proportion of training data to use for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load train and test data (data is already split)
    train_dataset = ADNIDataset(data_dir, split='train')
    test_dataset = ADNIDataset(data_dir, split='test')

    # Split training data into train and validation
    train_size = int((1 - val_split) * len(train_dataset))

    indices = list(range(len(train_dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader