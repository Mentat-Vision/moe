# blip_server.py
import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime
import time
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import threading

# Suppress warnings and verbose output
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def load_config():
    """Load configuration from config.env"""
    config = {
        "server_ip": "0.0.0.0",
        "port": 5001,
        "model_name": "Salesforce/blip-image-captioning-base",
        "processing_interval": 2.0,
        "use_gpu": True,
        "cuda_device": "cuda"
    }
    
    if os.path.exists("../config.env"):
        with open("../config.env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "SERVER_IP":
                        config["server_ip"] = value
                    elif key == "BLIP_PORT":
                        config["port"] = int(value)
                    elif key == "BLIP_MODEL_NAME":
                        config["model_name"] = value
                    elif key == "BLIP_PROCESSING_INTERVAL":
                        config["processing_interval"] = float(value)
                    elif key == "USE_GPU":
                        config["use_gpu"] = value.lower() == "true"
                    elif key == "CUDA_DEVICE":
                        config["cuda_device"] = value
    
    return config

class BLIPWebSocketServer:
    def __init__(self):
        # Load configuration
        self.config = load_config()
        
        # Initialize connected clients set
        self.connected_clients = set()
        
        # Load BLIP model
        self.processor = BlipProcessor.from_pretrained(self.config["model_name"])
        self.model = BlipForConditionalGeneration.from_pretrained(self.config["model_name"])
        
        # Move to GPU if available and enabled
        if self.config["use_gpu"] and torch.cuda.is_available():
            self.model = self.model.to(self.config["cuda_device"])
            print("🚀 BLIP model loaded on GPU")
        else:
            print("🚀 BLIP model loaded on CPU")
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.model_size = "BLIP-Base"
        
        # Performance tracking
        self.last_detection_time = time.time()
        self.processing_interval = self.config["processing_interval"]
        
        print("📝 BLIP WebSocket Server initialized")
        print(f"⚙️  Configuration: Port={self.config['port']}, GPU={self.config['use_gpu']}")
        
    async def handle_client(self, websocket, path):
        """Handle individual client connection"""
        self.connected_clients.add(websocket)
        client_address = websocket.remote_address
        print(f"🔌 Client connected: {client_address}")
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Process frame with BLIP
                        results = await self.process_frame(frame)
                        
                        # Send results back to client
                        await websocket.send(json.dumps(results))
                    else:
                        await websocket.send(json.dumps({"error": "Invalid frame data"}))
                else:
                    await websocket.send(json.dumps({"error": "Expected binary frame data"}))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {client_address}")
        except Exception as e:
            print(f"❌ Error handling client {client_address}: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def process_frame(self, frame):
        """Process frame with BLIP model"""
        try:
            current_time = time.time()
            
            # Control processing rate
            if current_time - self.last_detection_time < self.processing_interval:
                return {
                    "caption": "",
                    "fps": self.fps,
                    "frame_count": self.frame_count,
                    "model_size": self.model_size
                }
            
            self.last_detection_time = current_time
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process image with BLIP
            inputs = self.processor(frame_rgb, return_tensors="pt")
            
            # Move inputs to GPU if available and enabled
            if self.config["use_gpu"] and torch.cuda.is_available():
                inputs = {k: v.to(self.config["cuda_device"]) for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Update FPS
            self.frame_count += 1
            elapsed_time = current_time - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time
            
            return {
                "caption": caption,
                "fps": round(self.fps, 2),
                "frame_count": self.frame_count,
                "model_size": self.model_size
            }
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return {
                "error": str(e),
                "caption": "",
                "fps": self.fps,
                "frame_count": self.frame_count,
                "model_size": self.model_size
            }

    async def run_server(self):
        """Run the WebSocket server"""
        server = await websockets.serve(
            self.handle_client,
            self.config["server_ip"],
            self.config["port"]
        )
        
        print(f"🚀 BLIP WebSocket Server running on {self.config['server_ip']}:{self.config['port']}")
        print(f"📊 Connected clients: {len(self.connected_clients)}")
        
        await server.wait_closed()

async def main():
    server = BLIPWebSocketServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main()) 