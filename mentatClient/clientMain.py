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

def load_config():
    """Load configuration from config.env"""
    config = {
        "ENABLE_WINDOW_PREVIEW": True,  # Default to True
        "SERVER_IP": "10.8.162.58",
        "SERVER_PORT": "5000"
    }
    
    if os.path.exists("config.env"):
        with open("config.env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line or "=" not in line:
                    continue
                
                try:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove inline comments (everything after #)
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    
                    if key == "ENABLE_WINDOW_PREVIEW":
                        config[key] = value.lower() == "true"
                    elif key in ["SERVER_IP", "SERVER_PORT"]:
                        config[key] = value
                        
                except ValueError:
                    print(f"❌ Invalid configuration line: {line}")
                    continue
    
    return config

def get_enabled_cameras():
    """Get list of enabled cameras from config.env"""
    cameras = {}
    
    if not os.path.exists("config.env"):
        # Default: enable camera 0 and 1
        return {"webcam_0": 0, "webcam_1": 1}
    
    with open("config.env", "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith("#") or not line or "=" not in line:
                continue
            
            # Parse camera configuration lines
            if line.startswith("CAMERA_"):
                try:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove inline comments (everything after #)
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    
                    # Extract camera name (remove CAMERA_ prefix)
                    camera_name = key[7:]  # Remove "CAMERA_" prefix
                    
                    # Determine if it's a webcam index or RTSP URL
                    if value.startswith("rtsp://"):
                        cameras[camera_name] = value
                    else:
                        # Try to parse as integer for webcam index
                        try:
                            cameras[camera_name] = int(value)
                        except ValueError:
                            print(f"❌ Invalid camera value for {key}: {value}")
                            continue
                            
                except ValueError:
                    print(f"❌ Invalid camera configuration line: {line}")
                    continue
    
    if not cameras:
        print("ℹ️ No cameras enabled in config.env. Using default webcams 0 and 1")
        return {"webcam_0": 0, "webcam_1": 1}
    
    print(f"📹 Enabled cameras: {list(cameras.keys())}")
    return cameras

class MultiCameraClient:
    def __init__(self):
        # Load configuration
        self.config = load_config()
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
        
        # Processing scale (will be updated from server)
        self.processing_scale = 0.5
        
        # Initialize data structures for each camera
        for camera_name in self.cameras:
            self.yolo_data[camera_name] = {
                "detections": [],
                "person_detections": [],
                "person_count": 0,
                "fps": 0
            }
            self.blip_data[camera_name] = {
                "caption": "",
                "fps": 0
            }
            self.connected[camera_name] = False
            self.last_yolo_time[camera_name] = 0
            self.last_blip_time[camera_name] = 0
            self.camera_status[camera_name] = {"working": True, "failures": 0}
        
        # Print window preview status
        if self.config["ENABLE_WINDOW_PREVIEW"]:
            print("🖥️ Window preview: ENABLED")
        else:
            print("🖥️ Window preview: DISABLED (web streaming still active)")
        
        # Start listening for resolution updates
        self.start_resolution_listener()
    
    async def connect_to_server(self, camera_name):
        """Connect to central WebSocket server for specific camera"""
        try:
            # Use config values
            server_ip = self.config["SERVER_IP"]
            server_port = self.config["SERVER_PORT"]
            
            server_url = f"ws://{server_ip}:{server_port}"
            self.websockets[camera_name] = await websockets.connect(server_url)
            self.connected[camera_name] = True
            print(f"🔌 Camera {camera_name} connected to server: {server_url}")
            return True
        except Exception as e:
            print(f"❌ Camera {camera_name} failed to connect to server: {e}")
            return False
    
    def open_camera(self, camera_name, camera_source):
        """Open camera (webcam or RTSP stream)"""
        try:
            cap = cv2.VideoCapture(camera_source)
            
            # Get client preview scale from config
            preview_scale = float(self.config.get("CLIENT_PREVIEW_SCALE", 0.5))
            
            # Set properties for better performance
            if isinstance(camera_source, int):
                # Webcam settings - use scale to calculate target resolution
                # Assume 1920x1080 as base resolution for webcams
                base_width, base_height = 1920, 1080
                target_width = int(base_width * preview_scale)
                target_height = int(base_height * preview_scale)
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                # RTSP settings - keep original resolution but optimize buffering
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 25)
                # Don't force resolution for RTSP - let it use native resolution
            
            if not cap.isOpened():
                print(f"❌ Failed to open camera {camera_name} ({camera_source})")
                return None
            
            # Get actual resolution
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ Camera {camera_name} opened successfully ({width}x{height})")
            return cap
            
        except Exception as e:
            print(f"❌ Error opening camera {camera_name}: {e}")
            return None
    
    async def send_frame_to_expert(self, camera_name, frame, expert_type):
        """Send frame to specific expert through central server"""
        if not self.connected[camera_name] or camera_name not in self.websockets:
            return
        
        try:
            # Send frame at original resolution - server will handle scaling
            # This ensures client and server are in sync
            frame_resized = frame  # No resizing on client side
            
            # Encode frame as base64
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create message with expert type and camera info
            message = {
                "expert": expert_type,
                "camera_id": camera_name,  # Use camera name as ID
                "frame": frame_base64
            }
            
            # Send message
            await self.websockets[camera_name].send(json.dumps(message))
            
            # Wait for response
            timeout = 5.0 if expert_type == "BLIP" else 2.0
            response = await asyncio.wait_for(self.websockets[camera_name].recv(), timeout=timeout)
            results = json.loads(response)
            
            # Handle response based on expert type
            if expert_type == "YOLO" and "error" not in results:
                self.yolo_data[camera_name]["detections"] = results.get("detections", [])
                self.yolo_data[camera_name]["person_detections"] = results.get("person_detections", [])
                self.yolo_data[camera_name]["person_count"] = results.get("person_count", 0)
                self.yolo_data[camera_name]["fps"] = results.get("fps", 0)
                
                if self.yolo_data[camera_name]["detections"]:
                    labels = [f"{d['class']} ({d['confidence']:.2f})" for d in self.yolo_data[camera_name]["detections"]]
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"🎯 Camera {camera_name} - {timestamp} - {', '.join(labels)} (FPS: {self.yolo_data[camera_name]['fps']}, Persons: {self.yolo_data[camera_name]['person_count']})")
                    
            elif expert_type == "BLIP" and "error" not in results:
                self.blip_data[camera_name]["caption"] = results.get("caption", "")
                self.blip_data[camera_name]["fps"] = results.get("fps", 0)
                
                if self.blip_data[camera_name]["caption"]:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"📝 Camera {camera_name} - {timestamp} - {self.blip_data[camera_name]['caption']} (FPS: {self.blip_data[camera_name]['fps']})")
                    
            elif "error" in results:
                print(f"❌ Camera {camera_name} {expert_type} error: {results['error']}")
                    
        except asyncio.TimeoutError:
            print(f"⏰ Camera {camera_name} {expert_type} timeout")
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Camera {camera_name} connection closed, attempting to reconnect...")
            self.connected[camera_name] = False
            # Try to reconnect
            await self.connect_to_server(camera_name)
        except Exception as e:
            print(f"❌ Camera {camera_name} {expert_type} error: {e}")
    
    def draw_yolo_detections(self, frame, camera_name):
        """Draw YOLO detections on frame"""
        detections = self.yolo_data[camera_name]["detections"]
        
        # Use current processing scale from server
        processing_scale = self.processing_scale
        
        # Get current frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # The bounding boxes were calculated on frames scaled by processing_scale
        # We need to scale them from the processing size to the display size
        # The display frame is also scaled by client preview scale
        preview_scale = float(self.config.get("CLIENT_PREVIEW_SCALE", 0.5))
        scale_x = (1.0 / processing_scale) * preview_scale
        scale_y = (1.0 / processing_scale) * preview_scale
        
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            color = self.colors[i % len(self.colors)]
            
            # Scale bounding box coordinates to match display frame size
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_person_ids(self, frame, camera_name):
        """Draw person IDs on bounding boxes"""
        person_detections = self.yolo_data[camera_name]["person_detections"]
        
        # Use current processing scale from server
        processing_scale = self.processing_scale
        
        # Get current frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # The bounding boxes were calculated on frames scaled by processing_scale
        # We need to scale them from the processing size to the display size
        # The display frame is also scaled by client preview scale
        preview_scale = float(self.config.get("CLIENT_PREVIEW_SCALE", 0.5))
        scale_x = (1.0 / processing_scale) * preview_scale
        scale_y = (1.0 / processing_scale) * preview_scale
        
        for person in person_detections:
            bbox = person["bbox"]
            
            # Only draw ID if it exists in the detection
            if "id" in person:
                person_id = person["id"]
                
                # Scale bounding box coordinates to match display frame size
                x1 = int(bbox[0] * scale_x)
                y1 = int(bbox[1] * scale_y)
                x2 = int(bbox[2] * scale_x)
                y2 = int(bbox[3] * scale_y)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                id_text = f"ID: {person_id}"
                cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    def draw_blip_caption(self, frame, camera_name):
        """Draw BLIP caption on frame"""
        caption = self.blip_data[camera_name]["caption"]
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
    
    def draw_status_info(self, frame, camera_name):
        """Draw status information on frame"""
        # Position status info on the right side to avoid overlap
        frame_width = frame.shape[1]
        x_position = frame_width - 200  # 200px from right edge
        
        # Camera info
        cv2.putText(frame, f"Camera: {camera_name}", (x_position, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Connection status
        status_text = "Connected" if self.connected[camera_name] else "Disconnected"
        status_color = (0, 255, 0) if self.connected[camera_name] else (0, 0, 255)
        cv2.putText(frame, f"Server: {status_text}", (x_position, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, status_color, 2)
        
        # FPS info
        y_pos = 80
        if self.yolo_data[camera_name]["fps"] > 0:
            cv2.putText(frame, f"YOLO FPS: {self.yolo_data[camera_name]['fps']}", (x_position, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
        if self.blip_data[camera_name]["fps"] > 0:
            cv2.putText(frame, f"BLIP FPS: {self.blip_data[camera_name]['fps']}", (x_position, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
        
        # Person count
        if self.yolo_data[camera_name]["person_count"] > 0:
            cv2.putText(frame, f"Persons: {self.yolo_data[camera_name]['person_count']}", (x_position, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    async def run_async(self):
        """Main async loop"""
        # Connect to server for each camera
        for camera_name in self.cameras:
            await self.connect_to_server(camera_name)
        
        # Initialize video captures
        caps = {}
        for camera_name, camera_source in self.cameras.items():
            cap = self.open_camera(camera_name, camera_source)
            if cap is None:
                self.camera_status[camera_name]["working"] = False
                continue
            
            caps[camera_name] = cap
        
        if not caps:
            print("❌ No cameras could be opened. Check your configuration.")
            return
        
        print("🎥 Multi-Camera Client running with central server architecture.")
        if self.config["ENABLE_WINDOW_PREVIEW"]:
            print("Press 'q' to quit.")
        else:
            print("Running in headless mode (no window preview).")
            print("Press Ctrl+C to quit.")
        
        while True:
            current_time = time.time()
            
            # Process each camera
            for camera_name in self.cameras:
                if camera_name not in caps or not self.camera_status[camera_name]["working"]:
                    continue
                    
                cap = caps[camera_name]
                
                ret, frame = cap.read()
                if not ret:
                    self.camera_status[camera_name]["failures"] += 1
                    if self.camera_status[camera_name]["failures"] > 10:
                        print(f"❌ Camera {camera_name} failed too many times, disabling")
                        self.camera_status[camera_name]["working"] = False
                        cap.release()
                        del caps[camera_name]
                    continue
                
                # Reset failure count on successful read
                self.camera_status[camera_name]["failures"] = 0
                
                # Send frames at controlled rates
                if current_time - self.last_yolo_time[camera_name] >= self.yolo_interval:
                    await self.send_frame_to_expert(camera_name, frame, "YOLO")
                    self.last_yolo_time[camera_name] = current_time
                
                if current_time - self.last_blip_time[camera_name] >= self.blip_interval:
                    await self.send_frame_to_expert(camera_name, frame, "BLIP")
                    self.last_blip_time[camera_name] = current_time
                
                # Only draw overlays and show windows if preview is enabled
                if self.config["ENABLE_WINDOW_PREVIEW"]:
                    # Draw overlays
                    self.draw_yolo_detections(frame, camera_name)
                    self.draw_person_ids(frame, camera_name)
                    self.draw_blip_caption(frame, camera_name)
                    self.draw_status_info(frame, camera_name)
                    
                    # Resize frame for display using client preview scale
                    # Get client preview scale from config
                    preview_scale = float(self.config.get("CLIENT_PREVIEW_SCALE", 0.5))
                    
                    # Calculate display dimensions based on scale
                    display_width = int(frame.shape[1] * preview_scale)
                    display_height = int(frame.shape[0] * preview_scale)
                    display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    
                    # Show window
                    cv2.imshow(f"Camera {camera_name}", display_frame)
            
            # Handle quit key only if window preview is enabled
            if self.config["ENABLE_WINDOW_PREVIEW"]:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # In headless mode, just sleep a bit to prevent busy waiting
                await asyncio.sleep(0.01)
        
        # Cleanup
        for cap in caps.values():
            cap.release()
        
        if self.config["ENABLE_WINDOW_PREVIEW"]:
            cv2.destroyAllWindows()
        
        # Close WebSocket connections
        for websocket in self.websockets.values():
            await websocket.close()
    
    def start_resolution_listener(self):
        """Start listening for resolution updates from server"""
        try:
            import threading
            import requests
            
            def listen_for_updates():
                """Background thread to listen for resolution updates"""
                while True:
                    try:
                        # Poll server for resolution updates
                        response = requests.get(f"http://{self.config['SERVER_IP']}:5002/api/resolution/current", 
                                             timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            self.update_resolution_settings(data)
                    except Exception as e:
                        pass  # Silent fail for background polling
                    
                    time.sleep(10)  # Check every 10 seconds
            
            # Start background thread
            thread = threading.Thread(target=listen_for_updates, daemon=True)
            thread.start()
            print("📡 Resolution listener started")
            
        except Exception as e:
            print(f"❌ Error starting resolution listener: {e}")
    
    def update_resolution_settings(self, settings):
        """Update resolution settings from server"""
        try:
            if 'PROCESSING_SCALE' in settings:
                self.processing_scale = float(settings['PROCESSING_SCALE'])
                print(f"🔧 Processing scale updated: {self.processing_scale}")
                
        except Exception as e:
            print(f"❌ Error updating resolution settings: {e}")

def main():
    try:
        client = MultiCameraClient()
        asyncio.run(client.run_async())
    except ValueError as e:
        print(f"❌ {e}")
        print("💡 To enable cameras, edit config.env and uncomment the cameras you want to use.")
    except KeyboardInterrupt:
        print("\n🛑 Client stopped by user (Ctrl+C)")

if __name__ == "__main__":
    main()