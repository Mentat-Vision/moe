import cv2
import threading
import time
import requests
from io import BytesIO
import os
from typing import Dict, List

class CameraStream:
    def __init__(self, camera_id: str, camera_url: str, name: str):
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.name = name
        self.cap = None
        self.is_running = False
        self.frame = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start capturing from camera"""
        try:
            if self.camera_url.isdigit():
                # Webcam
                self.cap = cv2.VideoCapture(int(self.camera_url))
            else:
                # RTSP camera
                self.cap = cv2.VideoCapture(self.camera_url)
                
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.name} ({self.camera_url})")
                return False
                
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            print(f"Started camera {self.name}")
            return True
            
        except Exception as e:
            print(f"Error starting camera {self.name}: {e}")
            return False
    
    def _capture_loop(self):
        """Capture frames in a loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error capturing from {self.name}: {e}")
                time.sleep(1)
    
    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop capturing"""
        self.is_running = False
        if self.cap:
            self.cap.release()

class LocalClient:
    def __init__(self, server_url: str = None):
        # Get server URL from environment variable or use default
        if server_url is None:
            server_url = os.getenv('MENTAT_SERVER_URL', 'http://10.8.162.58:5000')
        
        self.server_url = server_url
        self.cameras: Dict[str, CameraStream] = {}
        self.is_running = False
        
        print(f"Connecting to server: {self.server_url}")
        
        # Camera configuration
        self.camera_config = {
            "CAMERA_0": "0",
            "CAMERA_1": "1", 
            "CAMERA_2": "2",
            # Uncomment and configure RTSP cameras as needed
            # "CAMERA_RTSP_101": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/101",
            # "CAMERA_RTSP_201": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/201",
            # "CAMERA_RTSP_301": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/301",
            # "CAMERA_RTSP_401": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/401",
            # "CAMERA_RTSP_501": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/501",
            # "CAMERA_RTSP_601": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/601",
        }
    
    def test_server_connection(self):
        """Test if server is reachable"""
        try:
            response = requests.get(f"{self.server_url}/api/cameras", timeout=5)
            if response.status_code == 200:
                print(f"✓ Server connection successful")
                return True
            else:
                print(f"✗ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to server: {e}")
            print(f"  Make sure the server is running and accessible at: {self.server_url}")
            return False
    
    def initialize_cameras(self):
        """Initialize all configured cameras"""
        for camera_id, camera_url in self.camera_config.items():
            camera_name = camera_id.replace("CAMERA_", "")
            camera = CameraStream(camera_id, camera_url, camera_name)
            if camera.start():
                self.cameras[camera_id] = camera
    
    def stream_to_server(self):
        """Stream camera feeds to server"""
        while self.is_running:
            try:
                for camera_id, camera in self.cameras.items():
                    frame = camera.get_frame()
                    if frame is not None:
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_data = buffer.tobytes()
                        
                        # Send to server
                        try:
                            response = requests.post(
                                f"{self.server_url}/stream/{camera_id}",
                                data=frame_data,
                                headers={'Content-Type': 'image/jpeg'},
                                timeout=2
                            )
                            if response.status_code != 200:
                                print(f"Failed to send frame from {camera.name}: {response.status_code}")
                        except requests.exceptions.RequestException as e:
                            print(f"Error sending frame from {camera.name}: {e}")
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Error in streaming loop: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the local client"""
        print("Initializing cameras...")
        self.initialize_cameras()
        
        if not self.cameras:
            print("No cameras available. Exiting.")
            return
        
        print(f"Started {len(self.cameras)} cameras")
        
        # Test server connection before streaming
        if not self.test_server_connection():
            print("Server connection failed. Please check:")
            print("1. Server is running on the correct IP and port")
            print("2. Network connectivity between client and server")
            print("3. Firewall settings allow the connection")
            print(f"4. Set MENTAT_SERVER_URL environment variable if needed")
            return
        
        print("Streaming to server...")
        
        self.is_running = True
        self.stream_to_server()
    
    def stop(self):
        """Stop the local client"""
        self.is_running = False
        for camera in self.cameras.values():
            camera.stop()

def main():
    # Allow command line argument for server URL
    import sys
    server_url = None
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    client = LocalClient(server_url)
    try:
        client.start()
    except KeyboardInterrupt:
        print("\nStopping client...")
        client.stop()

if __name__ == "__main__":
    main() 