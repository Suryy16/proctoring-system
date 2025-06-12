import os
import cv2
import numpy as np
from tqdm import tqdm

class VideoFrameExtractor:
    def __init__(self):
        pass

    def extract_uniform_frames(self, video_path, num_frames=20):
        """Extract frames uniformly spaced throughout the video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            # If video has fewer frames than requested, get all frames
            frame_indices = range(total_frames)
        else:
            # Calculate evenly spaced frame indices
            interval = max(total_frames // num_frames, 1)
            frame_indices = [i * interval for i in range(num_frames)]
            # Make sure we don't go beyond video length
            frame_indices = [min(i, total_frames-1) for i in frame_indices]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                # If couldn't read specific frame, try to get next available
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    
        cap.release()
        return frames

    def process_videos(self, input_dir, output_dir, num_frames=20):
        """Process all videos in directory and save uniformly spaced frames"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created at: {output_dir}")
        
        frame_paths = []
        labels = []
        
        for person_name in os.listdir(input_dir):
            person_path = os.path.join(input_dir, person_name)
            if not os.path.isdir(person_path):
                continue
                
            person_output_path = os.path.join(output_dir, person_name)
            os.makedirs(person_output_path, exist_ok=True)
            print(f"\nProcessing person: {person_name}")
            
            video_files = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            for video_name in tqdm(video_files, desc="Processing videos"):
                video_path = os.path.join(person_path, video_name)
                print(f"\nExtracting frames from: {video_name}")
                
                frames = self.extract_uniform_frames(video_path, num_frames)
                if not frames:
                    print(f"Warning: No frames extracted from {video_name}")
                    continue
                    
                # Save frames
                base_name = os.path.splitext(video_name)[0]
                for i, frame in enumerate(frames):
                    frame_name = f"{base_name}_frame_{i:04d}.jpg"
                    frame_path = os.path.join(person_output_path, frame_name)
                    if cv2.imwrite(frame_path, frame):
                        frame_paths.append(frame_path)
                        labels.append(person_name)
                        print(f"Saved: {frame_name}")
                    else:
                        print(f"Error saving: {frame_name}")
        
        print(f"\nCompleted processing. Total frames saved: {len(frame_paths)}")
        return frame_paths, labels

if __name__ == "__main__":
    extractor = VideoFrameExtractor()
    frame_paths, labels = extractor.process_videos("database/raw_data", "database/dataset", num_frames=28)