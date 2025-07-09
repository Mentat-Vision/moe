#!/usr/bin/env python3
"""
Model download script for the surveillance system.
Run this to download required models locally.
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
    print("🚀 Setting up surveillance system models...")
    
    # YOLO models (choose one based on your needs)
    yolo_models = {
        "modelsYolo/yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "modelsYolo/yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "modelsYolo/yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "modelsYolo/yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "modelsYolo/yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    }
    
    # Download YOLO models (you can choose which ones you need)
    for filename, url in yolo_models.items():
        if not os.path.exists(filename):
            download_file(url, filename)
        else:
            print(f"✅ {filename} already exists")
    
    print("\n📝 Note: For the Llama model, you'll need to download it manually:")
    print("   - Visit: https://huggingface.co/TheBloke/Llama-3.2-1B-Instruct-GGUF")
    print("   - Download: llama-3.2-1b-instruct-q4_k_m.gguf")
    print("   - Place it in: modelsChat/")
    
    print("\n🎉 Setup complete! You can now run the surveillance system.")

if __name__ == "__main__":
    main() 