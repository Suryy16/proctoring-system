import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

class DatasetProcessor:
    def __init__(self):
        self.model = YOLO("model/yolov11n-face.pt")

    def detect_faces(self, image):
        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(rgb_image)
        
        # Extract face bounding boxes
        faces = []
        for result in results:
            for box in result.boxes:
                if self.model.names[int(box.cls)] == 'face':  # or 'face' if your model is trained for faces
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    faces.append((x1, y1, x2 - x1, y2 - y1))  # (x, y, w, h)
        
        return faces
    
    def detect(self, image):
        """Detect faces in an image and return bounding boxes"""
        # Convert to RGB if needed
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(image, verbose=False)
        
        # Extract bounding boxes
        boxes = []
        for result in results:
            for box in result.boxes:
                # Filter only faces (class 0)
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    boxes.append([x1, y1, x2, y2, conf])
        
        return np.array(boxes) if boxes else np.zeros((0, 5))
    
    def extract_faces(self, image, padding=0.2):
        """Extract and return face regions"""
        detections = self.detect(image)
        faces = []
        
        for det in detections:
            x1, y1, x2, y2, _ = det
            h, w = image.shape[:2]
            
            # Add padding
            x1 = max(0, x1 - (x2 - x1) * padding)
            y1 = max(0, y1 - (y2 - y1) * padding)
            x2 = min(w, x2 + (x2 - x1) * padding)
            y2 = min(h, y2 + (y2 - y1) * padding)
            
            face = image[int(y1):int(y2), int(x1):int(x2)]
            if face.size > 0:  # Only add if face region is valid
                faces.append(face)
            
        return faces

    def process_dataset(self, dataset_path, output_path):
        """Process the dataset and extract faces, maintaining folder structure"""
        os.makedirs(output_path, exist_ok=True)
        
        faces = []
        labels = []
        
        # Iterate through each person's folder
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_path):
                continue
                
            # Create corresponding output folder
            person_output_path = os.path.join(output_path, person_name)
            os.makedirs(person_output_path, exist_ok=True)
            
            # Process each image for this person
            processed_count = 0
            for img_name in tqdm(os.listdir(person_path), desc=f"Processing {person_name}"):
                img_path = os.path.join(person_path, img_name)
                
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Could not read image: {img_path}")
                        continue
                        
                    detected_faces = self.extract_faces(image)
                    print(f"Found {len(detected_faces)} faces in {img_name}")
                    
                    for i, face in enumerate(detected_faces):
                        if face.size == 0:
                            print(f"Empty face detected in {img_name}")
                            continue
                            
                        face_filename = f"{os.path.splitext(img_name)[0]}_{processed_count}.jpg"
                        face_path = os.path.join(person_output_path, face_filename)
                        
                        if cv2.imwrite(face_path, face):
                            processed_count += 1
                            faces.append(face)
                            labels.append(person_name)
                        else:
                            print(f"Failed to save face: {face_path}")
                            
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

            print(f"Total processed for {person_name}: {processed_count}")
                    
        return faces, labels
    
if __name__ == "__main__":
    data_processor = DatasetProcessor()
    data_processor.process_dataset("database/dataset", "database/processed_dataset")