import cv2
import asyncio
import websockets
import requests
import json
import base64
import numpy as np
from datetime import datetime
import time

CAMERA_INDEX = 1
YOLO_SERVER_URL = "ws://10.8.162.58:5000"
BLIP_SERVER_URL = "http://10.8.162.58:5001/caption"

class UnifiedClient:
    def __init__(self):
        self.websocket = None
        self.connected = False
        self.detections = []
        self.person_detections = []
        self.person_count = 0
        self.fps = 0
        self.frame_count = 0
        self.model_size = "Unknown"
        self.caption = ""
        self.last_caption_time = 0
        self.caption_interval = 2.0  # Update caption every 2 seconds
        
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
        
    async def connect_yolo(self):
        """Connect to YOLO WebSocket server"""
        try:
            self.websocket = await websockets.connect(YOLO_SERVER_URL)
            self.connected = True
            print(f"🔌 Connected to YOLO server: {YOLO_SERVER_URL}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to YOLO: {e}")
            return False

    async def send_frame_to_yolo(self, frame):
        """Send frame to YOLO server and get detections"""
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
                
        except asyncio.TimeoutError:
            print("⏰ YOLO detection timeout")
        except Exception as e:
            print(f"❌ Error processing YOLO frame: {e}")
            self.connected = False

    def send_frame_to_blip(self, frame):
        """Send frame to BLIP server and get caption"""
        try:
            # Compress and encode frame
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
            
            # Send to BLIP server
            data = {"image": frame_base64}
            response = requests.post(BLIP_SERVER_URL, json=data, timeout=5)
            response.raise_for_status()
            
            result = response.json()
            self.caption = result.get("caption", "")
            print(f"📝 BLIP Caption: {self.caption}")
            
        except Exception as e:
            print(f"❌ Error processing BLIP frame: {e}")

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
            person_id = person["id"]
            
            # Draw person bounding box in red
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw ID on bounding box
            id_text = f"ID: {person_id}"
            cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_overlays(self, frame):
        """Draw all overlays on frame"""
        # Connection status
        status_text = f"YOLO: {'Connected' if self.connected else 'Disconnected'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0) if self.connected else (0, 0, 255), 2)
        
        # FPS and model info
        if self.fps > 0:
            fps_text = f"FPS: {self.fps}"
            cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            model_text = f"Model: {self.model_size}"
            cv2.putText(frame, model_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            person_count_text = f"Persons: {self.person_count}"
            cv2.putText(frame, person_count_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # BLIP caption
        if self.caption:
            # Create background for caption text
            caption_lines = self.caption.split('\n')
            y_offset = 160
            
            for line in caption_lines:
                if line.strip():
                    # Get text size for background
                    (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw background rectangle
                    cv2.rectangle(frame, (10, y_offset - text_height - 5), 
                                 (10 + text_width + 10, y_offset + 5), (0, 0, 0), -1)
                    
                    # Draw caption text
                    cv2.putText(frame, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 0), 2)  # Yellow text
                    y_offset += 30

    async def run_async(self):
        """Async main loop"""
        if not await self.connect_yolo():
            return
            
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("🎥 Unified Client running. Press 'q' to quit.")
        print("📊 YOLO: Real-time object detection")
        print("📝 BLIP: Image captioning")
        
        last_send_time = 0
        last_caption_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time()
            
            # Send frame to YOLO at controlled rate
            if current_time - last_send_time >= self.processing_interval:
                await self.send_frame_to_yolo(frame)
                last_send_time = current_time
            
            # Send frame to BLIP at slower rate
            if current_time - last_caption_time >= self.caption_interval:
                # Run BLIP in thread to avoid blocking
                import threading
                blip_thread = threading.Thread(target=self.send_frame_to_blip, args=(frame,))
                blip_thread.start()
                last_caption_time = current_time
            
            # Draw all detections and overlays
            self.draw_detections(frame)
            self.draw_person_ids(frame)
            self.draw_overlays(frame)
            
            cv2.imshow("Unified Analysis (YOLO + BLIP)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.websocket:
            await self.websocket.close()

def main():
    client = UnifiedClient()
    asyncio.run(client.run_async())

if __name__ == "__main__":
    main()