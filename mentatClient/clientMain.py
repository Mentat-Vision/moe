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
        
        # WebSocket connections for each camera
        self.yolo_websockets = {}
        self.blip_websockets = {}
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
        self.yolo_interval = 0.1  # 100ms between YOLO detections
        self.blip_interval = 2.0  # 2 seconds between BLIP captions
        
        # Camera status tracking
        self.camera_status = {}
        
        # Initialize data structures for each camera
        for camera_index in self.cameras:
            self.yolo_data[camera_index] = {
                "detections": [],
                "person_detections": [],
                "person_count": 0,
                "fps": 0,
                "model_size": "Unknown"
            }
            self.blip_data[camera_index] = {
                "caption": "",
                "fps": 0
            }
            self.connected[camera_index] = {"yolo": False, "blip": False}
            self.last_yolo_time[camera_index] = 0
            self.last_blip_time[camera_index] = 0
            self.camera_status[camera_index] = {"working": True, "failures": 0}
    
    async def connect_yolo(self, camera_index):
        """Connect to YOLO server for specific camera"""
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
            self.yolo_websockets[camera_index] = await websockets.connect(server_url)
            self.connected[camera_index]["yolo"] = True
            print(f"🔌 Camera {camera_index} connected to YOLO server: {server_url}")
            return True
        except Exception as e:
            print(f"❌ Camera {camera_index} failed to connect to YOLO: {e}")
            return False
    
    async def connect_blip(self, camera_index):
        """Connect to BLIP server for specific camera"""
        try:
            # Read server URL from config.env
            server_ip = "10.8.162.58"
            blip_port = "5001"
            
            if os.path.exists("config.env"):
                with open("config.env", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("SERVER_IP="):
                            server_ip = line.split("=", 1)[1]
                        elif line.startswith("BLIP_PORT="):
                            blip_port = line.split("=", 1)[1]
            
            server_url = f"ws://{server_ip}:{blip_port}"
            self.blip_websockets[camera_index] = await websockets.connect(server_url)
            self.connected[camera_index]["blip"] = True
            print(f"🔌 Camera {camera_index} connected to BLIP server: {server_url}")
            return True
        except Exception as e:
            print(f"❌ Camera {camera_index} failed to connect to BLIP: {e}")
            return False
    
    async def send_yolo_frame(self, camera_index, frame):
        """Send frame to YOLO server for specific camera"""
        if not self.connected[camera_index]["yolo"] or camera_index not in self.yolo_websockets:
            return
        
        try:
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            await self.yolo_websockets[camera_index].send(frame_bytes)
            response = await asyncio.wait_for(self.yolo_websockets[camera_index].recv(), timeout=2.0)
            results = json.loads(response)
            
            if "error" not in results:
                self.yolo_data[camera_index]["detections"] = results.get("detections", [])
                self.yolo_data[camera_index]["person_detections"] = results.get("person_detections", [])
                self.yolo_data[camera_index]["person_count"] = results.get("person_count", 0)
                self.yolo_data[camera_index]["fps"] = results.get("fps", 0)
                self.yolo_data[camera_index]["model_size"] = results.get("model_size", "Unknown")
                
                if self.yolo_data[camera_index]["detections"]:
                    labels = [f"{d['class']} ({d['confidence']:.2f})" for d in self.yolo_data[camera_index]["detections"]]
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"🎯 Camera {camera_index} - {timestamp} - {', '.join(labels)} (FPS: {self.yolo_data[camera_index]['fps']}, Persons: {self.yolo_data[camera_index]['person_count']})")
                    
        except asyncio.TimeoutError:
            print(f"⏰ Camera {camera_index} YOLO detection timeout")
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Camera {camera_index} YOLO connection closed, attempting to reconnect...")
            self.connected[camera_index]["yolo"] = False
            # Try to reconnect
            await self.connect_yolo(camera_index)
        except Exception as e:
            print(f"❌ Camera {camera_index} YOLO error: {e}")
            self.connected[camera_index]["yolo"] = False
    
    async def send_blip_frame(self, camera_index, frame):
        """Send frame to BLIP server for specific camera"""
        if not self.connected[camera_index]["blip"] or camera_index not in self.blip_websockets:
            print(f"🔍 Camera {camera_index} BLIP: Not connected or no websocket")
            return
        
        try:
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            print(f"🔍 Camera {camera_index} BLIP: Sending frame ({len(frame_bytes)} bytes)")
            await self.blip_websockets[camera_index].send(frame_bytes)
            response = await asyncio.wait_for(self.blip_websockets[camera_index].recv(), timeout=5.0)
            results = json.loads(response)
            
            if "error" not in results:
                self.blip_data[camera_index]["caption"] = results.get("caption", "")
                self.blip_data[camera_index]["fps"] = results.get("fps", 0)
                
                if self.blip_data[camera_index]["caption"]:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"📝 Camera {camera_index} - {timestamp} - {self.blip_data[camera_index]['caption']} (FPS: {self.blip_data[camera_index]['fps']})")
                else:
                    print(f"🔍 Camera {camera_index} BLIP: Empty caption received")
            else:
                print(f"🔍 Camera {camera_index} BLIP: Error in response: {results.get('error')}")
                    
        except asyncio.TimeoutError:
            print(f"⏰ Camera {camera_index} BLIP caption timeout")
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Camera {camera_index} BLIP connection closed, attempting to reconnect...")
            self.connected[camera_index]["blip"] = False
            # Try to reconnect
            await self.connect_blip(camera_index)
        except Exception as e:
            print(f"❌ Camera {camera_index} BLIP error: {e}")
            self.connected[camera_index]["blip"] = False
    
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
            
            y_position = 30
            for i, line in enumerate(lines[:3]):
                (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (10, y_position - text_height - 5), 
                            (10 + text_width + 10, y_position + 5), (0, 0, 0), -1)
                cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y_position += 25
    
    def draw_status_info(self, frame, camera_index):
        """Draw status information on frame"""
        # Camera info
        cv2.putText(frame, f"Camera: {camera_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Connection status
        yolo_status = "YOLO: Connected" if self.connected[camera_index]["yolo"] else "YOLO: Disconnected"
        blip_status = "BLIP: Connected" if self.connected[camera_index]["blip"] else "BLIP: Disconnected"
        
        cv2.putText(frame, yolo_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0) if self.connected[camera_index]["yolo"] else (0, 0, 255), 2)
        cv2.putText(frame, blip_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0) if self.connected[camera_index]["blip"] else (0, 0, 255), 2)
        
        # FPS info
        if self.yolo_data[camera_index]["fps"] > 0:
            cv2.putText(frame, f"YOLO FPS: {self.yolo_data[camera_index]['fps']}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.blip_data[camera_index]["fps"] > 0:
            cv2.putText(frame, f"BLIP FPS: {self.blip_data[camera_index]['fps']}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Person count
        if self.yolo_data[camera_index]["person_count"] > 0:
            cv2.putText(frame, f"Persons: {self.yolo_data[camera_index]['person_count']}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    async def run_async(self):
        """Main async loop"""
        # Connect to servers for each camera
        for camera_index in self.cameras:
            await self.connect_yolo(camera_index)
            await self.connect_blip(camera_index)
        
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
        
        print("🎥 Multi-Camera Client running with both YOLO and BLIP on each camera.")
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
                    await self.send_yolo_frame(camera_index, frame)
                    self.last_yolo_time[camera_index] = current_time
                
                if current_time - self.last_blip_time[camera_index] >= self.blip_interval:
                    print(f"🔍 Camera {camera_index}: Sending frame to BLIP (interval: {current_time - self.last_blip_time[camera_index]:.2f}s)")
                    await self.send_blip_frame(camera_index, frame)
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
        for websocket in self.yolo_websockets.values():
            await websocket.close()
        for websocket in self.blip_websockets.values():
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