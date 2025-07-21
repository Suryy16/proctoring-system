import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class HybridProctoringModel(nn.Module):
    def __init__(self, num_classes=2, tabular_input_dim=4):
        super(HybridProctoringModel, self).__init__()
        
        # Branch 1: CNN untuk image features (menggunakan ResNet18 backbone)
        self.image_backbone = models.resnet18(pretrained=True)
        # Modifikasi layer pertama untuk grayscale/RGB
        self.image_backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove final classification layer
        self.image_backbone.fc = nn.Identity()
        
        # Image feature processing
        self.image_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Branch 2: MLP untuk tabular features (HR, VR, Pitch, Yaw)
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Fusion layer - gabungkan image + tabular features
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + 16, 64),  # 128 from image + 16 from tabular
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, image, tabular):
        # Process image
        image_features = self.image_backbone(image)
        image_features = self.image_fc(image_features)
        
        # Process tabular data
        tabular_features = self.tabular_fc(tabular)
        
        # Fusion
        combined_features = torch.cat([image_features, tabular_features], dim=1)
        output = self.fusion_fc(combined_features)
        
        return output