"""
Dataset loaders for CBIS-DDSM and BreakHis datasets
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import logging

from config import Config

logger = logging.getLogger(__name__)


class BaseBreastDataset(Dataset):
    """Base class for breast imaging datasets"""
    
    def __init__(self, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.metadata = []
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'path': str(image_path),
            'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
        }


class CBISDDSMDataset(BaseBreastDataset):
    """
    CBIS-DDSM Dataset Loader
    
    CBIS-DDSM (Curated Breast Imaging Subset of DDSM) is a mammography dataset
    containing calcification and mass cases with pathology labels.
    
    Expected structure:
    cbis_ddsm/
        ├── manifest.csv (or metadata file)
        ├── Mass-Training/
        ├── Mass-Test/
        ├── Calc-Training/
        └── Calc-Test/
    """
    
    def __init__(self, root_dir: Path, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Root directory of CBIS-DDSM dataset
            split: 'train' or 'test'
            transform: Optional transform to apply to images
        """
        super().__init__(transform)
        self.root_dir = Path(root_dir)
        self.split = split
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset files and labels"""
        logger.info(f"Loading CBIS-DDSM {self.split} dataset from {self.root_dir}")
        
        if not self.root_dir.exists():
            logger.warning(f"CBIS-DDSM directory not found: {self.root_dir}")
            logger.warning("Please download from: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM")
            return
        
        # Look for training/test directories
        search_dirs = []
        if self.split == 'train':
            search_dirs = ['Mass-Training', 'Calc-Training']
        else:
            search_dirs = ['Mass-Test', 'Calc-Test']
        
        for search_dir in search_dirs:
            dir_path = self.root_dir / search_dir
            if dir_path.exists():
                # Recursively find all image files
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.dcm']:
                    image_files = list(dir_path.rglob(ext))
                    for img_path in image_files:
                        self.images.append(img_path)
                        
                        # Extract label from directory structure
                        # Typical structure: .../Benign_Without_Callback/... or .../Malignant/...
                        path_parts = img_path.parts
                        if 'Malignant' in path_parts or 'malignant' in str(img_path).lower():
                            label = 1  # Malignant
                        else:
                            label = 0  # Benign
                        
                        self.labels.append(label)
                        self.metadata.append({
                            'dataset': 'cbis_ddsm',
                            'type': 'Mass' if 'Mass' in search_dir else 'Calc',
                            'split': self.split
                        })
        
        logger.info(f"Loaded {len(self.images)} images from CBIS-DDSM {self.split} set")
        if len(self.images) > 0:
            logger.info(f"Label distribution: {np.bincount(self.labels)}")


class BreakHisDataset(BaseBreastDataset):
    """
    BreakHis Dataset Loader
    
    BreakHis (Breast Cancer Histopathological Database) contains microscopic
    images of breast tumor tissue collected from 82 patients.
    
    Expected structure:
    breakhis/
        ├── BreaKHis_v1/
        │   ├── histology_slides/
        │   │   └── breast/
        │   │       ├── benign/
        │   │       │   └── SOB/
        │   │       │       ├── adenosis/
        │   │       │       ├── fibroadenoma/
        │   │       │       ├── phyllodes_tumor/
        │   │       │       └── tubular_adenoma/
        │   │       └── malignant/
        │   │           └── SOB/
        │   │               ├── ductal_carcinoma/
        │   │               ├── lobular_carcinoma/
        │   │               ├── mucinous_carcinoma/
        │   │               └── papillary_carcinoma/
    """
    
    def __init__(self, root_dir: Path, split: str = 'train', 
                 magnification: Optional[str] = None, transform=None):
        """
        Args:
            root_dir: Root directory of BreakHis dataset
            split: 'train' or 'test'
            magnification: Specific magnification to use ('40X', '100X', '200X', '400X')
                          If None, uses all magnifications
            transform: Optional transform to apply to images
        """
        super().__init__(transform)
        self.root_dir = Path(root_dir)
        self.split = split
        self.magnification = magnification
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset files and labels"""
        logger.info(f"Loading BreakHis {self.split} dataset from {self.root_dir}")
        
        if not self.root_dir.exists():
            logger.warning(f"BreakHis directory not found: {self.root_dir}")
            logger.warning("Please download from: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/")
            return
        
        # Look for the standard BreakHis structure
        histology_path = self.root_dir / "BreaKHis_v1" / "histology_slides" / "breast"
        if not histology_path.exists():
            histology_path = self.root_dir  # Try direct path
        
        for label_name, label_value in [('benign', 0), ('malignant', 1)]:
            label_path = histology_path / label_name
            if not label_path.exists():
                continue
            
            # Find all image files
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files = list(label_path.rglob(ext))
                for img_path in image_files:
                    # Filter by magnification if specified
                    if self.magnification and self.magnification not in str(img_path):
                        continue
                    
                    self.images.append(img_path)
                    self.labels.append(label_value)
                    
                    # Extract subtype from path
                    path_str = str(img_path)
                    subtype = 'unknown'
                    for st in ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma',
                              'ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 
                              'papillary_carcinoma']:
                        if st in path_str:
                            subtype = st
                            break
                    
                    # Extract magnification from filename
                    mag = 'unknown'
                    for m in ['40X', '100X', '200X', '400X']:
                        if m in str(img_path.name):
                            mag = m
                            break
                    
                    self.metadata.append({
                        'dataset': 'breakhis',
                        'subtype': subtype,
                        'magnification': mag,
                        'split': self.split
                    })
        
        # Split train/test (80/20 split if not already split)
        if len(self.images) > 0:
            indices = list(range(len(self.images)))
            np.random.seed(42)
            np.random.shuffle(indices)
            
            split_idx = int(0.8 * len(indices))
            if self.split == 'train':
                indices = indices[:split_idx]
            else:
                indices = indices[split_idx:]
            
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.metadata = [self.metadata[i] for i in indices]
        
        logger.info(f"Loaded {len(self.images)} images from BreakHis {self.split} set")
        if len(self.images) > 0:
            logger.info(f"Label distribution: {np.bincount(self.labels)}")


class CombinedBreastDataset(Dataset):
    """Combined dataset from CBIS-DDSM and BreakHis"""
    
    def __init__(self, cbis_dataset: CBISDDSMDataset, breakhis_dataset: BreakHisDataset):
        self.cbis_dataset = cbis_dataset
        self.breakhis_dataset = breakhis_dataset
        self.cbis_len = len(cbis_dataset)
        self.breakhis_len = len(breakhis_dataset)
        self.total_len = self.cbis_len + self.breakhis_len
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.cbis_len:
            return self.cbis_dataset[idx]
        else:
            return self.breakhis_dataset[idx - self.cbis_len]


def get_transforms(split: str = 'train'):
    """Get image transforms for training or validation"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ])


def create_dataloaders(batch_size: int = None) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for training and validation
    
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    # Create datasets
    cbis_train = CBISDDSMDataset(
        Config.CBIS_DDSM_DIR, 
        split='train', 
        transform=get_transforms('train')
    )
    cbis_val = CBISDDSMDataset(
        Config.CBIS_DDSM_DIR, 
        split='test', 
        transform=get_transforms('val')
    )
    
    breakhis_train = BreakHisDataset(
        Config.BREAKHIS_DIR, 
        split='train', 
        transform=get_transforms('train')
    )
    breakhis_val = BreakHisDataset(
        Config.BREAKHIS_DIR, 
        split='test', 
        transform=get_transforms('val')
    )
    
    # Combine datasets
    train_dataset = CombinedBreastDataset(cbis_train, breakhis_train)
    val_dataset = CombinedBreastDataset(cbis_val, breakhis_val)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }
