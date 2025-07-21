import torch
import joblib
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from .hybrid_model import HybridProctoringModel

class HybridPredictor:
    def __init__(self, model_path='model/hybrid_model.pth', scaler_path='model/hybrid_scaler.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = HybridProctoringModel(num_classes=2, tabular_input_dim=4)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, frame):
        """Convert OpenCV frame to PIL Image and apply transforms"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image
    
    def predict(self, frame, hr, vr, pitch, yaw):
        """
        Predict cheating probability from frame and tabular features
        
        Args:
            frame: OpenCV image frame
            hr: Horizontal ratio
            vr: Vertical ratio 
            pitch: Head pose pitch
            yaw: Head pose yaw
            
        Returns:
            prediction: 0=Normal, 1=Cheating
            probabilities: [prob_normal, prob_cheating]
        """
        # Preprocess image
        image = self.preprocess_image(frame).to(self.device)
        
        # Preprocess tabular features
        tabular_features = np.array([[hr, vr, pitch, yaw]], dtype=np.float32)
        tabular_features = self.scaler.transform(tabular_features)
        tabular_features = torch.tensor(tabular_features, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image, tabular_features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()[0], probabilities.cpu().numpy()[0]