import os
import glob
from torch.utils.data import Dataset

class ADNIDataset(Dataset):
    """
    Dataset class for ADNI brain MRI classification
    Two classes: AD (Alzheimer's Disease) and NC (Normal Control)
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

    def __len__(self):
        return len(self.image_files)