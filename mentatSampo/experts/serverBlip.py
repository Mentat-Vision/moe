# blip_server.py
import cv2
import numpy as np
import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from .baseWorker import BaseWorker

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class BLIPWorker(BaseWorker):
    """BLIP expert worker that processes image captioning jobs"""
    
    def __init__(self, config):
        super().__init__("BLIP", config)
        self.model = None
        self.processor = None
        self.device = "cpu"
    
    async def initialize_model(self):
        """Initialize the BLIP model"""
        model_name = self.config.get("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
        use_gpu = self.config.get("USE_GPU", "true").lower() == "true"
        cuda_device = self.config.get("CUDA_DEVICE", "cuda")
        
        try:
            # Load BLIP model and processor
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            # Move to GPU if available and enabled
            if use_gpu and torch.cuda.is_available():
                self.device = cuda_device
                self.model = self.model.to(self.device)
                print(f"✅ BLIP model loaded on GPU: {model_name}")
            else:
                self.device = "cpu"
                print(f"✅ BLIP model loaded on CPU: {model_name}")
                
        except Exception as e:
            print(f"❌ Error loading BLIP model: {e}")
            raise e
    
    async def process_frame(self, job):
        """Process a frame with BLIP image captioning"""
        try:
            frame = job["frame"]
            camera_id = job["camera_id"]
            
            if self.model is None or self.processor is None:
                return {"error": "BLIP model not loaded"}
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process image with BLIP
            inputs = self.processor(frame_rgb, return_tensors="pt")
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Get current stats
            stats = self.get_stats()
            
            return {
                "caption": caption,
                "fps": stats["fps"],
                "camera_id": camera_id
            }
            
        except Exception as e:
            print(f"❌ BLIP Worker error processing frame: {e}")
            return {
                "error": str(e),
                "caption": "",
                "fps": 0,
                "camera_id": job.get("camera_id", 0)
            } 