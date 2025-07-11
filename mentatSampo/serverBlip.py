# blip_server.py
from flask import Flask, request, jsonify
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import cv2
import numpy as np
import warnings
import os
import gc

# Suppress warnings and verbose output
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_grad_enabled(False)  # Disable gradients globally

app = Flask(__name__)

class BLIPServer:
    def __init__(self):
        # Optimized BLIP configuration
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use half precision for speed
            low_cpu_mem_usage=True
        )

        # Optimized device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()

        # Memory optimization
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def process_image(self, image_data):
        """Process image and return caption"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Optimized frame processing
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Process with optimized settings
            inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=25,  # Balanced length for detail
                    num_beams=3,    # Small beam search for better quality
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.2,  # Prevent repetitive text
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return caption
        except Exception as e:
            return f"Error processing image: {str(e)}"

# Initialize BLIP server
blip_server = BLIPServer()

@app.route('/caption', methods=['POST'])
def caption():
    data = request.get_json()
    image_data = data.get("image", "")
    
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400
    
    caption = blip_server.process_image(image_data)
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001) 