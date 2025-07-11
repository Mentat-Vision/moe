import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime
import threading
import time
import os

# ===== CAMERA CONFIGURATION =====
# Change this to use different camera (0 or 1)
CAMERA_INDEX = 0
# =================================

def get_enabled_cameras():
    """Get list of enabled camera indices from config.env"""
    enabled_cameras = []
    
    if not os.path.exists("config.env"):
        # Default: enable camera 0 and 1
        return [0, 1]
    
    with open("config.env", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("CAMERAS="):
                cameras_str = line.split("=", 1)[1]
                try:
                    # Parse comma-separated camera indices
                    cameras = [int(c.strip()) for c in cameras_str.split(",") if c.strip()]
                    enabled_cameras = cameras
                except (ValueError, IndexError):
                    print("❌ Invalid CAMERAS format in config.env. Using default cameras [0, 1]")
                    enabled_cameras = [0, 1]
                break
    
    return enabled_cameras if enabled_cameras else [0, 1]  # Default to camera 0 and 1

class YOLOWebSocketClient:
    def __init__(self):
        # Get enabled cameras
        enabled_cameras = get_enabled_cameras()
        
        if CAMERA_INDEX not in enabled_cameras:
            raise ValueError(f"Camera {CAMERA_INDEX} not enabled. Check config.env file.")
        
        self.camera_index = CAMERA_INDEX
        self.websocket = None
        self.connected = False
        self.detections = []
        self.person_detections = []
        self.person_count = 0
        self.fps = 0
        self.frame_count = 0
        self.model_size = "Unknown"
        
        # Performance tracking
        self.last_detection_time = time.time()
        self.processing_interval = 0.1  # 100ms between detections
        
        # Color palette for different bounding boxes
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 0, 0),    # Dark Blue
            (0, 0, 128),    # Dark Red
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (255, 20, 147), # Deep Pink
        ]
        
    async def connect(self):
        """Connect to YOLO WebSocket server"""
        try:
            # Read server URL from config.env
            server_ip = "10.8.162.58"
            yolo_port = "5000"
            
            if os.path.exists("config.env"):
                with open("config.env", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("SERVER_IP="):
                            server_ip = line.split("=", 1)[1]
                        elif line.startswith("YOLO_PORT="):
                            yolo_port = line.split("=", 1)[1]
            
            server_url = f"ws://{server_ip}:{yolo_port}"
            self.websocket = await websockets.connect(server_url)
            self.connected = True
            print(f"🔌 Connected to YOLO server: {server_url}")
            print(f"📷 Using camera index: {self.camera_index}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    async def send_frame(self, frame):
        """Send frame to server and get detections"""
        if not self.connected or not self.websocket:
            return
            
        try:
            # Compress and encode frame
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Send frame as binary data
            await self.websocket.send(frame_bytes)
            
            # Receive detection results
            response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
            results = json.loads(response)
            
            if "error" not in results:
                self.detections = results.get("detections", [])
                self.person_detections = results.get("person_detections", [])
                self.person_count = results.get("person_count", 0)
                self.fps = results.get("fps", 0)
                self.frame_count = results.get("frame_count", 0)
                self.model_size = results.get("model_size", "Unknown")
                
                # Log detections
                if self.detections:
                    labels = [f"{d['class']} ({d['confidence']:.2f})" for d in self.detections]
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"{timestamp} - {', '.join(labels)} (FPS: {self.fps}, Model: {self.model_size}, Persons: {self.person_count})")
                    
        except asyncio.TimeoutError:
            print("⏰ Detection timeout")
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            self.connected = False

    def draw_detections(self, frame):
        """Draw detection boxes and labels on frame with different colors"""
        for i, detection in enumerate(self.detections):
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            # Get color for this detection (cycle through colors)
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background rectangle for text (using same color as box)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), color, -1)
            
            # Text (white for good contrast)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_person_ids(self, frame):
        """Draw person IDs on bounding boxes"""
        for person in self.person_detections:
            bbox = person["bbox"]
            
            # Only draw ID if it exists in the detection
            if "id" in person:
                person_id = person["id"]
                
                # Draw person bounding box in red
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw ID on bounding box
                id_text = f"ID: {person_id}"
                cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)

    async def run_async(self):
        """Async main loop"""
        if not await self.connect():
            return
            
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"🎥 YOLO WebSocket Client running on camera {self.camera_index}. Press 'q' to quit.")
        
        last_send_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to read from camera {self.camera_index}")
                break
                
            current_time = time.time()
            
            # Send frame at controlled rate
            if current_time - last_send_time >= self.processing_interval:
                await self.send_frame(frame)
                last_send_time = current_time
            
            # Draw detections on frame
            self.draw_detections(frame)
            
            # Draw person IDs
            self.draw_person_ids(frame)
            
            # Display camera info
            camera_text = f"Camera: {self.camera_index}"
            cv2.putText(frame, camera_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # Display connection status
            status_text = f"Connected: {'Yes' if self.connected else 'No'}"
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0) if self.connected else (0, 0, 255), 2)
            
            # Display FPS and model size
            if self.fps > 0:
                fps_text = f"FPS: {self.fps}"
                cv2.putText(frame, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                model_text = f"Model: {self.model_size}"
                cv2.putText(frame, model_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                person_count_text = f"Persons: {self.person_count}"
                cv2.putText(frame, person_count_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
            
            cv2.imshow(f"YOLO - Camera {self.camera_index}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.websocket:
            await self.websocket.close()

def main():
    try:
        client = YOLOWebSocketClient()
        asyncio.run(client.run_async())
    except ValueError as e:
        print(f"❌ {e}")
        print("📷 Available cameras:")
        for camera_index in get_enabled_cameras():
            print(f"  Camera {camera_index}")

if __name__ == "__main__":
    main() 