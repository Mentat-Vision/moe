import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class YOLOWebSocketServer:
    def __init__(self):
        # Load YOLO11x model
        self.model = YOLO("./modelsYolo/yolo11l.pt") 
        # n, fps=8
        # s, fps=8
        # m, fps=8
        
        # Extract model size from filename
        model_path = "./modelsYolo/yolo11l.pt"
        self.model_size = model_path.split('/')[-1].replace('.pt', '').replace('yolo11', '')
        print(f"🚀 YOLO model loaded successfully: {self.model_size.upper()}")
        
        # Performance tracking
        self.frame_count = 0
        self.avg_fps = 0
        self.last_time = datetime.now()
        
        # Person tracking variables
        self.person_tracks = {}  # Store person positions for ID assignment
        self.next_person_id = 1

    def assign_person_ids(self, results):
        """Assign IDs to detected persons based on position"""
        person_detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[int(cls_id)]
                
                if class_name == "person":
                    # Calculate center of bounding box
                    x1, y1, x2, y2 = box
                    center_x = float((x1 + x2) / 2)
                    center_y = float((y1 + y2) / 2)
                    
                    # Find closest existing person track
                    min_distance = float('inf')
                    assigned_id = None
                    
                    for person_id, (track_x, track_y) in self.person_tracks.items():
                        distance = ((center_x - track_x) ** 2 + (center_y - track_y) ** 2) ** 0.5
                        if distance < min_distance and distance < 100:  # Threshold for same person
                            min_distance = distance
                            assigned_id = person_id
                    
                    # If no close match found, assign new ID
                    if assigned_id is None:
                        assigned_id = self.next_person_id
                        self.next_person_id += 1
                    
                    # Update track
                    self.person_tracks[assigned_id] = (center_x, center_y)
                    
                    person_detections.append({
                        'id': assigned_id,
                        'bbox': [float(x) for x in box.tolist()],
                        'confidence': float(conf),
                        'center': [center_x, center_y]
                    })
        
        return person_detections

    async def process_frame(self, frame_data):
        """Process frame and return detection results"""
        try:
            # Decode base64 frame
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)[0]
            
            # Extract detection data
            detections = []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    class_name = self.model.names[int(cls_id)]
                    detections.append({
                        "class": class_name,
                        "confidence": float(conf),
                        "bbox": [float(x) for x in box.tolist()]  # [x1, y1, x2, y2]
                    })
            
            # Get person detections with IDs
            person_detections = self.assign_person_ids(results)
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = datetime.now()
                elapsed = (current_time - self.last_time).total_seconds()
                self.avg_fps = 30 / elapsed if elapsed > 0 else 0
                self.last_time = current_time
            
            return {
                "detections": detections,
                "person_detections": person_detections,
                "person_count": len(person_detections),
                "fps": round(self.avg_fps, 1),
                "frame_count": self.frame_count,
                "model_size": self.model_size.upper()
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
    server = YOLOWebSocketServer()
    
    # Start WebSocket server
    start_server = websockets.serve(
        server.handle_client, 
        "0.0.0.0", 
        5000,  # Using port 5000 as requested
        max_size=10 * 1024 * 1024  # 10MB max message size
    )
    
    print("🚀 YOLO WebSocket Server starting on ws://0.0.0.0:5000")
    print("📊 Ready to process real-time YOLO detections")
    
    await start_server
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 