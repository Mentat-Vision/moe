import cv2
import asyncio
import websockets
import websockets.exceptions
import json
import base64
import numpy as np
from datetime import datetime
import time
import threading
import os

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

class MultiCameraClient:
    def __init__(self):
        self.cameras = get_enabled_cameras()
        
        if not self.cameras:
            raise ValueError("No cameras enabled. Check config.env file.")
        
        # Single WebSocket connection per camera
        self.websockets = {}
        self.connected = {}
        
        # Data storage for each camera
        self.yolo_data = {}
        self.blip_data = {}
        
        # Color palette for YOLO detections
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        # Performance tracking
        self.last_yolo_time = {}
        self.last_blip_time = {}
        self.yolo_interval = 0.2  # 200ms between YOLO detections (5 FPS)
        self.blip_interval = 3.0  # 3 seconds between BLIP captions
        
        # Camera status tracking
        self.camera_status = {}
        
        # Initialize data structures for each camera
        for camera_index in self.cameras:
            self.yolo_data[camera_index] = {
                "detections": [],
                "person_detections": [],
                "person_count": 0,
                "fps": 0
            }
            self.blip_data[camera_index] = {
                "caption": "",
                "fps": 0
            }
            self.connected[camera_index] = False
            self.last_yolo_time[camera_index] = 0
            self.last_blip_time[camera_index] = 0
            self.camera_status[camera_index] = {"working": True, "failures": 0}
    
    async def connect_to_server(self, camera_index):
        """Connect to central WebSocket server for specific camera"""
        try:
            # Read server URL from config.env
            server_ip = "10.8.162.58"
            server_port = "5000"
            
            if os.path.exists("config.env"):
                with open("config.env", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("SERVER_IP="):
                            server_ip = line.split("=", 1)[1]
                        elif line.startswith("SERVER_PORT="):
                            server_port = line.split("=", 1)[1]
            
            server_url = f"ws://{server_ip}:{server_port}"
            self.websockets[camera_index] = await websockets.connect(server_url)
            self.connected[camera_index] = True
            print(f"🔌 Camera {camera_index} connected to server: {server_url}")
            return True
        except Exception as e:
            print(f"❌ Camera {camera_index} failed to connect to server: {e}")
            return False
    
    async def send_frame_to_expert(self, camera_index, frame, expert_type):
        """Send frame to specific expert through central server"""
        if not self.connected[camera_index] or camera_index not in self.websockets:
            return
        
        try:
            # Resize frame for processing
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            
            # Encode frame as base64
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create message with expert type and camera info
            message = {
                "expert": expert_type,
                "camera_id": camera_index,
                "frame": frame_base64
            }
            
            # Send message
            await self.websockets[camera_index].send(json.dumps(message))
            
            # Wait for response
            timeout = 5.0 if expert_type == "BLIP" else 2.0
            response = await asyncio.wait_for(self.websockets[camera_index].recv(), timeout=timeout)
            results = json.loads(response)
            
            # Handle response based on expert type
            if expert_type == "YOLO" and "error" not in results:
                self.yolo_data[camera_index]["detections"] = results.get("detections", [])
                self.yolo_data[camera_index]["person_detections"] = results.get("person_detections", [])
                self.yolo_data[camera_index]["person_count"] = results.get("person_count", 0)
                self.yolo_data[camera_index]["fps"] = results.get("fps", 0)
                
                if self.yolo_data[camera_index]["detections"]:
                    labels = [f"{d['class']} ({d['confidence']:.2f})" for d in self.yolo_data[camera_index]["detections"]]
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"🎯 Camera {camera_index} - {timestamp} - {', '.join(labels)} (FPS: {self.yolo_data[camera_index]['fps']}, Persons: {self.yolo_data[camera_index]['person_count']})")
                    
            elif expert_type == "BLIP" and "error" not in results:
                self.blip_data[camera_index]["caption"] = results.get("caption", "")
                self.blip_data[camera_index]["fps"] = results.get("fps", 0)
                
                if self.blip_data[camera_index]["caption"]:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"📝 Camera {camera_index} - {timestamp} - {self.blip_data[camera_index]['caption']} (FPS: {self.blip_data[camera_index]['fps']})")
                    
            elif "error" in results:
                print(f"❌ Camera {camera_index} {expert_type} error: {results['error']}")
                    
        except asyncio.TimeoutError:
            print(f"⏰ Camera {camera_index} {expert_type} timeout")
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Camera {camera_index} connection closed, attempting to reconnect...")
            self.connected[camera_index] = False
            # Try to reconnect
            await self.connect_to_server(camera_index)
        except Exception as e:
            print(f"❌ Camera {camera_index} {expert_type} error: {e}")
    
    def draw_yolo_detections(self, frame, camera_index):
        """Draw YOLO detections on frame"""
        detections = self.yolo_data[camera_index]["detections"]
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            color = self.colors[i % len(self.colors)]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_person_ids(self, frame, camera_index):
        """Draw person IDs on bounding boxes"""
        person_detections = self.yolo_data[camera_index]["person_detections"]
        for person in person_detections:
            bbox = person["bbox"]
            
            # Only draw ID if it exists in the detection
            if "id" in person:
                person_id = person["id"]
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                id_text = f"ID: {person_id}"
                cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    def draw_blip_caption(self, frame, camera_index):
        """Draw BLIP caption on frame"""
        caption = self.blip_data[camera_index]["caption"]
        if caption:
            words = caption.split()
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
            
            # Position caption at bottom of frame to avoid overlap
            frame_height = frame.shape[0]
            y_position = frame_height - 80  # Start 80px from bottom
            
            for i, line in enumerate(lines[:3]):
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (10, y_position - text_height - 5), 
                            (10 + text_width + 10, y_position + 5), (0, 0, 0), -1)
                cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y_position += 25
    
    def draw_status_info(self, frame, camera_index):
        """Draw status information on frame"""
        # Position status info on the right side to avoid overlap
        frame_width = frame.shape[1]
        x_position = frame_width - 200  # 200px from right edge
        
        # Camera info
        cv2.putText(frame, f"Camera: {camera_index}", (x_position, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Connection status
        status_text = "Connected" if self.connected[camera_index] else "Disconnected"
        status_color = (0, 255, 0) if self.connected[camera_index] else (0, 0, 255)
        cv2.putText(frame, f"Server: {status_text}", (x_position, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, status_color, 2)
        
        # FPS info
        y_pos = 80
        if self.yolo_data[camera_index]["fps"] > 0:
            cv2.putText(frame, f"YOLO FPS: {self.yolo_data[camera_index]['fps']}", (x_position, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
        if self.blip_data[camera_index]["fps"] > 0:
            cv2.putText(frame, f"BLIP FPS: {self.blip_data[camera_index]['fps']}", (x_position, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
        
        # Person count
        if self.yolo_data[camera_index]["person_count"] > 0:
            cv2.putText(frame, f"Persons: {self.yolo_data[camera_index]['person_count']}", (x_position, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    async def run_async(self):
        """Main async loop"""
        # Connect to server for each camera
        for camera_index in self.cameras:
            await self.connect_to_server(camera_index)
        
        # Initialize video captures
        caps = {}
        for camera_index in self.cameras:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"❌ Failed to open camera {camera_index}")
                self.camera_status[camera_index]["working"] = False
                continue
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps[camera_index] = cap
            print(f"✅ Camera {camera_index} initialized successfully")
        
        print("🎥 Multi-Camera Client running with central server architecture.")
        print("Press 'q' to quit.")
        
        while True:
            current_time = time.time()
            
            # Process each camera
            for camera_index in self.cameras:
                if camera_index not in caps or not self.camera_status[camera_index]["working"]:
                    continue
                    
                cap = caps[camera_index]
                
                ret, frame = cap.read()
                if not ret:
                    self.camera_status[camera_index]["failures"] += 1
                    if self.camera_status[camera_index]["failures"] > 10:
                        print(f"❌ Camera {camera_index} failed too many times, disabling")
                        self.camera_status[camera_index]["working"] = False
                        cap.release()
                        del caps[camera_index]
                    continue
                
                # Reset failure count on successful read
                self.camera_status[camera_index]["failures"] = 0
                
                # Send frames at controlled rates
                if current_time - self.last_yolo_time[camera_index] >= self.yolo_interval:
                    await self.send_frame_to_expert(camera_index, frame, "YOLO")
                    self.last_yolo_time[camera_index] = current_time
                
                if current_time - self.last_blip_time[camera_index] >= self.blip_interval:
                    await self.send_frame_to_expert(camera_index, frame, "BLIP")
                    self.last_blip_time[camera_index] = current_time
                
                # Draw overlays
                self.draw_yolo_detections(frame, camera_index)
                self.draw_person_ids(frame, camera_index)
                self.draw_blip_caption(frame, camera_index)
                self.draw_status_info(frame, camera_index)
                
                # Show window
                cv2.imshow(f"Camera {camera_index}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        
        # Close WebSocket connections
        for websocket in self.websockets.values():
            await websocket.close()

def main():
    try:
        client = MultiCameraClient()
        asyncio.run(client.run_async())
    except ValueError as e:
        print(f"❌ {e}")
        print("💡 To enable cameras, edit config.env and uncomment the cameras you want to use.")

if __name__ == "__main__":
    main()