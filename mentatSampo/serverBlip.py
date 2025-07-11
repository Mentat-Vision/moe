# blip_server.py
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import warnings
import os
import gc

# Suppress warnings and verbose output
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_grad_enabled(False)  # Disable gradients globally

class BLIPWebSocketServer:
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
            
        # Performance tracking
        self.frame_count = 0
        self.avg_fps = 0
        self.last_time = datetime.now()
        
        print("🚀 BLIP model loaded successfully")

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

    async def process_frame(self, frame_data):
        """Process frame and return caption results"""
        try:
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = datetime.now()
                elapsed = (current_time - self.last_time).total_seconds()
                self.avg_fps = 30 / elapsed if elapsed > 0 else 0
                self.last_time = current_time
            
            # Process image and get caption
            caption = self.process_image(frame_data)
            
            return {
                "caption": caption,
                "fps": round(self.avg_fps, 1),
                "frame_count": self.frame_count
            }
            
        except Exception as e:
            return {"error": str(e)}

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        client_id = id(websocket)
        print(f" Client {client_id} connected")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    # Handle JSON messages (commands, etc.)
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                else:
                    # Handle binary frame data
                    frame_data = base64.b64encode(message).decode('utf-8')
                    results = await self.process_frame(frame_data)
                    await websocket.send(json.dumps(results))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f" Client {client_id} disconnected")
        except Exception as e:
            print(f"❌ Error handling client {client_id}: {e}")

async def main():
    server = BLIPWebSocketServer()
    
    # Start WebSocket server
    start_server = websockets.serve(
        server.handle_client, 
        "0.0.0.0", 
        5001,  # Using port 5001 as requested
        max_size=10 * 1024 * 1024  # 10MB max message size
    )
    
    print("🚀 BLIP WebSocket Server starting on ws://0.0.0.0:5001")
    print("📊 Ready to process real-time BLIP captions")
    
    await start_server
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 