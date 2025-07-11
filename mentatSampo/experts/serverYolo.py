import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime
import time
import os
from ultralytics import YOLO
import threading

def load_config():
    """Load configuration from config.env"""
    config = {
        "server_ip": "0.0.0.0",
        "port": 5000,
        "model_path": "modelsYolo/yolo11s.pt",
        "processing_interval": 0.1,
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
                    elif key == "YOLO_PORT":
                        config["port"] = int(value)
                    elif key == "YOLO_MODEL_PATH":
                        config["model_path"] = value
                    elif key == "YOLO_PROCESSING_INTERVAL":
                        config["processing_interval"] = float(value)
                    elif key == "USE_GPU":
                        config["use_gpu"] = value.lower() == "true"
                    elif key == "CUDA_DEVICE":
                        config["cuda_device"] = value
    
    return config

class YOLOWebSocketServer:
    def __init__(self):
        # Load configuration
        self.config = load_config()
        
        # Initialize connected clients set
        self.connected_clients = set()
        
        # Load YOLO model
        model_path = self.config["model_path"]
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("💡 Please download the YOLO model first")
            print(f"💡 Expected path: {os.path.abspath(model_path)}")
            print("❌ No YOLO model found. Please run modelsDownload.py first.")
            return
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return
        
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.model_size = "YOLO11L"
        
        # Performance tracking
        self.last_detection_time = time.time()
        self.processing_interval = self.config["processing_interval"]
        
        print(f"🎯 YOLO WebSocket Server initialized with model: {model_path}")
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
                        # Process frame with YOLO
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
        """Process frame with YOLO model"""
        try:
            current_time = time.time()
            
            # Control processing rate
            if current_time - self.last_detection_time < self.processing_interval:
                return {
                    "detections": [],
                    "person_detections": [],
                    "person_count": 0,
                    "fps": self.fps,
                    "frame_count": self.frame_count,
                    "model_size": self.model_size
                }
            
            self.last_detection_time = current_time
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Extract detections
            detections = []
            person_detections = []
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],  # Convert to Python float
                            "class": class_name,
                            "confidence": confidence,
                            "class_id": class_id
                        }
                        detections.append(detection)
                        
                        # Count persons
                        if class_name.lower() == "person":
                            person_count += 1
                            person_detections.append(detection)
            
            # Update FPS
            self.frame_count += 1
            elapsed_time = current_time - self.start_time
            if elapsed_time > 0:
                self.fps = self.frame_count / elapsed_time
            
            return {
                "detections": detections,
                "person_detections": person_detections,
                "person_count": person_count,
                "fps": round(self.fps, 2),
                "frame_count": self.frame_count,
                "model_size": self.model_size
            }
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return {
                "error": str(e),
                "detections": [],
                "person_detections": [],
                "person_count": 0,
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
        
        print(f"🚀 YOLO WebSocket Server running on {self.config['server_ip']}:{self.config['port']}")
        print(f"📊 Connected clients: {len(self.connected_clients)}")
        
        await server.wait_closed()

async def main():
    server = YOLOWebSocketServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main()) 