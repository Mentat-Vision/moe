import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime
import time

CAMERA_INDEX = 1
YOLO_SERVER_URL = "ws://10.8.162.58:5000"
BLIP_SERVER_URL = "ws://10.8.162.58:5001"  # Updated to WebSocket

class UnifiedClient:
    def __init__(self):
        # YOLO connection
        self.yolo_websocket = None
        self.yolo_connected = False
        self.detections = []
        self.person_detections = []
        self.person_count = 0
        self.yolo_fps = 0
        self.yolo_frame_count = 0
        self.yolo_model_size = "Unknown"
        
        # BLIP connection
        self.blip_websocket = None
        self.blip_connected = False
        self.caption = ""
        self.blip_fps = 0
        self.blip_frame_count = 0
        
        # Performance tracking
        self.last_yolo_time = time.time()
        self.last_blip_time = time.time()
        self.yolo_interval = 0.1  # 100ms between YOLO detections
        self.blip_interval = 2.0  # 2 seconds between BLIP captions
        
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
            self.yolo_websocket = await websockets.connect(YOLO_SERVER_URL)
            self.yolo_connected = True
            print(f"🔌 Connected to YOLO server: {YOLO_SERVER_URL}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to YOLO: {e}")
            return False

    async def connect_blip(self):
        """Connect to BLIP WebSocket server"""
        try:
            self.blip_websocket = await websockets.connect(BLIP_SERVER_URL)
            self.blip_connected = True
            print(f"🔌 Connected to BLIP server: {BLIP_SERVER_URL}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to BLIP: {e}")
            return False

    async def send_frame_to_yolo(self, frame):
        """Send frame to YOLO server and get detections"""
        if not self.yolo_connected or not self.yolo_websocket:
            return
            
        try:
            # Compress and encode frame
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Send frame as binary data
            await self.yolo_websocket.send(frame_bytes)
            
            # Receive detection results with shorter timeout
            response = await asyncio.wait_for(self.yolo_websocket.recv(), timeout=1.0)
            results = json.loads(response)
            
            if "error" not in results:
                self.detections = results.get("detections", [])
                self.person_detections = results.get("person_detections", [])
                self.person_count = results.get("person_count", 0)
                self.yolo_fps = results.get("fps", 0)
                self.yolo_frame_count = results.get("frame_count", 0)
                self.yolo_model_size = results.get("model_size", "Unknown")
                
        except asyncio.TimeoutError:
            print("⏰ YOLO detection timeout")
        except websockets.exceptions.ConnectionClosed:
            print("🔌 YOLO connection closed")
            self.yolo_connected = False
        except Exception as e:
            print(f"❌ Error processing YOLO frame: {e}")
            self.yolo_connected = False

    async def send_frame_to_blip(self, frame):
        """Send frame to BLIP server and get caption"""
        if not self.blip_connected or not self.blip_websocket:
            return
            
        try:
            # Compress and encode frame
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Send frame as binary data
            await self.blip_websocket.send(frame_bytes)
            
            # Receive caption results with longer timeout for BLIP
            response = await asyncio.wait_for(self.blip_websocket.recv(), timeout=10.0)
            results = json.loads(response)
            
            if "error" not in results:
                self.caption = results.get("caption", "")
                self.blip_fps = results.get("fps", 0)
                self.blip_frame_count = results.get("frame_count", 0)
                
                # Log caption
                if self.caption:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"{timestamp} - BLIP: {self.caption}")
                    
        except asyncio.TimeoutError:
            print("⏰ BLIP caption timeout")
        except websockets.exceptions.ConnectionClosed:
            print("🔌 BLIP connection closed")
            self.blip_connected = False
        except Exception as e:
            print(f"❌ Error processing BLIP frame: {e}")
            self.blip_connected = False

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

    def draw_caption_overlay(self, frame):
        """Draw BLIP caption overlay on frame"""
        if self.caption:
            # Word wrapping for better display
            words = self.caption.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) < 40:
                    current_line += (" " + word) if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Display up to 3 lines with black background
            y_position = 160
            for i, line in enumerate(lines[:3]):
                # Get text size to create background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw black background rectangle
                cv2.rectangle(frame, (10, y_position - text_height - 5), 
                            (10 + text_width + 10, y_position + 5), (0, 0, 0), -1)
                
                # Draw text on top of background (yellow for BLIP)
                cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y_position += 25

    def draw_overlays(self, frame):
        """Draw all overlays on frame"""
        # Connection status
        yolo_status = f"YOLO: {'Connected' if self.yolo_connected else 'Disconnected'}"
        blip_status = f"BLIP: {'Connected' if self.blip_connected else 'Disconnected'}"
        
        cv2.putText(frame, yolo_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0) if self.yolo_connected else (0, 0, 255), 2)
        cv2.putText(frame, blip_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0) if self.blip_connected else (0, 0, 255), 2)
        
        # YOLO info
        if self.yolo_fps > 0:
            yolo_fps_text = f"YOLO FPS: {self.yolo_fps}"
            cv2.putText(frame, yolo_fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            model_text = f"Model: {self.yolo_model_size}"
            cv2.putText(frame, model_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            person_count_text = f"Persons: {self.person_count}"
            cv2.putText(frame, person_count_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # BLIP info
        if self.blip_fps > 0:
            blip_fps_text = f"BLIP FPS: {self.blip_fps}"
            cv2.putText(frame, blip_fps_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)

    async def run_async(self):
        """Async main loop"""
        # Connect to both servers
        yolo_connected = await self.connect_yolo()
        blip_connected = await self.connect_blip()
        
        if not yolo_connected and not blip_connected:
            print("❌ Failed to connect to any servers")
            print("💡 Make sure both servers are running:")
            print("   • YOLO: python serverYolo.py")
            print("   • BLIP: python serverBlip.py")
            print("   • Or use: python serverMain.py")
            return
            
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("🎥 Unified Client running. Press 'q' to quit.")
        print("📊 YOLO: Real-time object detection")
        print("📝 BLIP: Image captioning")
        
        last_yolo_time = 0
        last_blip_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = time.time()
                
                # Send frame to YOLO at controlled rate
                if current_time - last_yolo_time >= self.yolo_interval:
                    await self.send_frame_to_yolo(frame)
                    last_yolo_time = current_time
                
                # Send frame to BLIP at controlled rate
                if current_time - last_blip_time >= self.blip_interval:
                    await self.send_frame_to_blip(frame)
                    last_blip_time = current_time
                
                # Draw all detections and overlays
                self.draw_detections(frame)
                self.draw_person_ids(frame)
                self.draw_caption_overlay(frame)
                self.draw_overlays(frame)
                
                cv2.imshow("Unified Analysis (YOLO + BLIP)", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n Received interrupt signal")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Close WebSocket connections
            if self.yolo_websocket:
                await self.yolo_websocket.close()
            if self.blip_websocket:
                await self.blip_websocket.close()

def main():
    client = UnifiedClient()
    asyncio.run(client.run_async())

if __name__ == "__main__":
    main()