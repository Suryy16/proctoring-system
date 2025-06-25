import cv2
import os
from deepface import DeepFace
import pandas as pd
from recognition_scripts.config import CONFIG

class FaceRecognizer:
    def process_face(img, face_region):
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        face_img = img[y:y+h, x:x+w]
        
        # Resize untuk konsistensi
        face_img = cv2.resize(face_img, (112, 112))
        
        return face_img

    def recognize_face(image, face_region, threshold=0.7):
        x, y, w, h = face_region
        face_img = image[y:y+h, x:x+w]
        
        try:
            # Find matching faces using Facenet512
            dfs = DeepFace.find(
                img_path=face_img,
                db_path=CONFIG["database_path"],
                model_name=CONFIG["model_name"],
                distance_metric="cosine",
                enforce_detection=False,
                silent=True
            )

            # Check if we got any results
            if not dfs or len(dfs) == 0:
                return "Unknown", 0.0
                
            df = dfs[0]  # Get the first DataFrame
            
            # Check if DataFrame is empty
            if not isinstance(df, pd.DataFrame) or df.empty:
                return "Unknown", 0.0
                
            # Find distance column - it might be called 'distance' or contain 'cosine'
            distance_columns = [col for col in df.columns if 'distance' in col.lower() or 'cosine' in col.lower()]
            if not distance_columns:
                # print("Logging: No distance column found")
                # print(f"Available columns: {list(df.columns)}")
                return "Unknown", 0.0
                
            distance_column = distance_columns[0]
            best_match = df.iloc[0]
            
            # Convert distance to similarity (cosine distance is already 0-2 where 0 is identical)
            similarity = 1 - (best_match[distance_column]/2)  # Normalize to 0-1 range
            
            # print(f"Logging: best match identity - {best_match['identity']}")
            # print(f"Logging: raw distance - {best_match[distance_column]:.4f}")
            # print(f"Logging: similarity score - {similarity:.2f}")
            
            if similarity >= threshold:
                identity = os.path.basename(os.path.dirname(best_match['identity']))
                return identity, similarity
                
            print(f"Logging: similarity below threshold ({similarity:.2f} < {threshold})")
            return "Unknown", 0.0
            
        except Exception as e:
            print(f"Recognition error: {str(e)}")
            return "Unknown", 0.0