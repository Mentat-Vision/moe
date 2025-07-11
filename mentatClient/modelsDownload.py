#!/usr/bin/env python3
"""
Model download script for the MOE system.
Downloads only the models actually used in the codebase.
"""

import os
import requests
from pathlib import Path

def download_file(url, filename):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
    
    print(f"\n✅ Downloaded {filename}")

def main():
    """Download required models"""
    print("🚀 Setting up MOE system models...")
    
    # YOLO models - only download what's actually used
    # The server uses yolo11l.pt, but we'll download a few options
    yolo_models = {
        "modelsYolo/yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "modelsYolo/yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "modelsYolo/yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "modelsYolo/yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11l.pt"
    }
    
    # Download YOLO models
    for filename, url in yolo_models.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"✅ {filename} already exists")
    
    print("\n📝 Note: BLIP models are downloaded automatically by transformers library")
    print("   when you first run the BLIP server.")
    
    print("\n🎉 Setup complete! You can now run the MOE system.")
    print("   - Run servers: python mentatSampo/serverMain.py")
    print("   - Run client: python mentatClient/clientMain.py")

if __name__ == "__main__":
    main() 