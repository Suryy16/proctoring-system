import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

class HybridProctoringDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, scaler=None, is_train=True):
        self.df = pd.read_csv(csv_path).dropna()
        self.image_dir = image_dir
        self.transform = transform if transform else self.get_default_transform()
        
        # Tabular features
        self.tabular_features = self.df[['HR', 'VR', 'Pitch', 'Yaw']].values.astype(np.float32)
        
        # Scaling tabular features
        if is_train and scaler is None:
            self.scaler = StandardScaler()
            self.tabular_features = self.scaler.fit_transform(self.tabular_features)
        elif scaler is not None:
            self.scaler = scaler
            self.tabular_features = self.scaler.transform(self.tabular_features)
        else:
            self.scaler = None
            
        # Labels
        if 'Label' in self.df.columns:
            self.labels = self.df['Label'].values.astype(np.int64)
        else:
            self.labels = None
            
        # Timestamps untuk mencari corresponding images
        self.timestamps = self.df['timestamp'].values
        
    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_image_path_from_timestamp(self, timestamp):
        # Implementasi untuk mencari image berdasarkan timestamp
        # Asumsi: image disimpan dengan nama berdasarkan timestamp
        timestamp_str = pd.to_datetime(timestamp).strftime("%Y%m%d_%H%M%S")
        possible_extensions = ['.jpg', '.jpeg', '.png']
        
        for ext in possible_extensions:
            image_path = os.path.join(self.image_dir, f"frame_{timestamp_str}{ext}")
            if os.path.exists(image_path):
                return image_path
        
        # Fallback: return dummy image or raise error
        return None
    
    def __len__(self):
        return len(self.tabular_features)
    
    def __getitem__(self, idx):
        # Get tabular features
        tabular_features = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        
        # Get corresponding image
        timestamp = self.timestamps[idx]
        image_path = self.get_image_path_from_timestamp(timestamp)
        
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            # Create dummy image if not found
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, tabular_features, label
        else:
            return image, tabular_features