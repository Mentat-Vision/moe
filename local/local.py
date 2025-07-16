# local.py
import cv2
import threading
import time
import socketio
import os
from typing import Dict
from collections import deque

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
        try:
            backend = cv2.CAP_FFMPEG if 'rtsp' in self.camera_url.lower() else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(self.camera_url) if self.camera_url.isdigit() else self.camera_url, backend)
            if not self.cap.isOpened():
                print(f"✗ Failed to open camera {self.name} ({self.camera_url})")
                return False

            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            print(f"✓ Started camera {self.name}")
            return True
        except Exception as e:
            print(f"Error starting camera {self.name}: {e}")
            return False

    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Resize for efficiency
                frame = cv2.resize(frame, (640, 360))  # Balanced resolution; adjust as needed
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

class LocalClient:
    def __init__(self, server_url=None):
        self.server_url = server_url or os.getenv('MENTAT_SERVER_URL', 'http://10.8.162.58:5000')
        self.sio = socketio.Client()
        self.cameras: Dict[str, CameraStream] = {}
        self.connected = False
        self.is_running = False

        # Define your camera sources here, comment/uncomment freely
        self.camera_config = {
            "CAMERA_0": "0",  # Mac webcam
            "CAMERA_1": "1",
            "CAMERA_2": "2",
            "CAMERA_RTSP_101": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/101",
            "CAMERA_RTSP_201": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/201",
            "CAMERA_RTSP_301": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/301",
            "CAMERA_RTSP_401": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/401",
            "CAMERA_RTSP_501": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/501",
            "CAMERA_RTSP_601": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/601",
        }

        self._setup_handlers()

    def _setup_handlers(self):
        @self.sio.event
        def connect():
            print("✓ Connected to server")
            self.connected = True
            for camera_id in self.cameras:
                self.sio.emit('register_camera', {
                    'camera_id': camera_id,
                    'name': self.cameras[camera_id].name
                })
        @self.sio.event
        def disconnect():
            print("✗ Disconnected from server")
            self.connected = False

        @self.sio.event
        def connected(data):
            print(f"✓ Server acknowledged: {data}")

        @self.sio.event
        def camera_registered(data):
            print(f"✓ Registered: {data}")

        @self.sio.event
        def frame_received(data):
            if data.get("status") != "success":
                print(f"✗ Frame failed: {data.get('message')}")

    def connect_to_server(self):
        try:
            self.sio.connect(self.server_url)
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def initialize_cameras(self):
        for camera_id, camera_url in self.camera_config.items():
            name = camera_id.replace("CAMERA_", "")
            cam = CameraStream(camera_id, camera_url, name)
            if cam.start():
                self.cameras[camera_id] = cam

    def stream_to_server(self):
        while self.is_running:
            if not self.connected and not self.connect_to_server():
                time.sleep(3)
                continue

            for camera_id, cam in self.cameras.items():
                frame = cam.get_frame()
                if frame is not None:
                    # Optimize for speed - lower quality but faster encoding
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                    try:
                        # Send binary bytes directly, no base64
                        self.sio.emit('frame', {
                            'camera_id': camera_id,
                            'frame': buf.tobytes()
                        })
                    except Exception as e:
                        print(f"Emit error: {e}")
                        self.connected = False
            time.sleep(0.025)  # Aim for ~40 FPS client-side, server will throttle to 30

    def start(self):
        self.initialize_cameras()
        if not self.cameras:
            print("No cameras started.")
            return
        self.is_running = True
        self.stream_to_server()

    def stop(self):
        self.is_running = False
        if self.connected:
            self.sio.disconnect()
        for cam in self.cameras.values():
            cam.stop()

def main():
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else None
    client = LocalClient(url)
    try:
        client.start()
    except KeyboardInterrupt:
        print("Stopping...")
        client.stop()

if __name__ == "__main__":
    main()