import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime
import time
import os

# ===== CAMERA CONFIGURATION =====
# Change this to use different camera (0 or 1)
CAMERA_INDEX = 1
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

class BLIPWebSocketClient:
    def __init__(self):
        # Get enabled cameras
        enabled_cameras = get_enabled_cameras()
        
        if CAMERA_INDEX not in enabled_cameras:
            raise ValueError(f"Camera {CAMERA_INDEX} not enabled. Check config.env file.")
        
        self.camera_index = CAMERA_INDEX
        self.websocket = None
        self.connected = False
        self.caption = ""
        self.fps = 0
        self.frame_count = 0
        
        # Performance tracking
        self.last_detection_time = time.time()
        self.processing_interval = 2.0  # 2 seconds between captions
        
    async def connect(self):
        """Connect to BLIP WebSocket server"""
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
            self.websocket = await websockets.connect(server_url)
            self.connected = True
            print(f"🔌 Connected to BLIP server: {server_url}")
            print(f"📷 Using camera index: {self.camera_index}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    async def send_frame(self, frame):
        """Send frame to server and get caption"""
        if not self.connected or not self.websocket:
            return
            
        try:
            # Compress and encode frame
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Send frame as binary data
            await self.websocket.send(frame_bytes)
            
            # Receive caption results
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            results = json.loads(response)
            
            if "error" not in results:
                self.caption = results.get("caption", "")
                self.fps = results.get("fps", 0)
                self.frame_count = results.get("frame_count", 0)
                
                # Log caption
                if self.caption:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"{timestamp} - {self.caption} (FPS: {self.fps})")
                    
        except asyncio.TimeoutError:
            print("⏰ Caption timeout")
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            self.connected = False

    def draw_caption_overlay(self, frame):
        """Draw caption overlay on frame"""
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
            y_position = 30
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

    async def run_async(self):
        """Async main loop"""
        if not await self.connect():
            return
            
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"🎥 BLIP WebSocket Client running on camera {self.camera_index}. Press 'q' to quit.")
        
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
            
            # Draw caption overlay
            self.draw_caption_overlay(frame)
            
            # Display camera info
            camera_text = f"Camera: {self.camera_index}"
            cv2.putText(frame, camera_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # Display connection status
            status_text = f"Connected: {'Yes' if self.connected else 'No'}"
            cv2.putText(frame, status_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0) if self.connected else (0, 0, 255), 2)
            
            # Display FPS
            if self.fps > 0:
                fps_text = f"FPS: {self.fps}"
                cv2.putText(frame, fps_text, (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
            
            cv2.imshow(f"BLIP - Camera {self.camera_index}", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.websocket:
            await self.websocket.close()

def main():
    try:
        client = BLIPWebSocketClient()
        asyncio.run(client.run_async())
    except ValueError as e:
        print(f"❌ {e}")
        print("📷 Available cameras:")
        for camera_index in get_enabled_cameras():
            print(f"  Camera {camera_index}")

if __name__ == "__main__":
    main() 