# local.py
import cv2
import threading
import time
import socketio
import os
from typing import Dict
import subprocess
import io
import hashlib  # For frame change detection
import datetime

class CameraStream:
    def __init__(self, camera_id: str, camera_url: str, name: str):
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.name = name
        self.cap = None
        self.process = None  # For FFmpeg subprocess
        self.is_running = False
        self.frame = None
        self.last_frame_hash = None  # For change detection
        self.lock = threading.Lock()

    def start(self):
        try:
            if 'rtsp' in self.camera_url.lower():
                # Use FFmpeg for RTSP sources
                self._start_ffmpeg()
            else:
                # Use OpenCV for non-RTSP (e.g., local webcams)
                backend = cv2.CAP_ANY
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

    def _start_ffmpeg(self):
        # FFmpeg command to capture RTSP and output JPEG frames to stdout
        cmd = [
            'ffmpeg',
            '-i', self.camera_url,
            '-f', 'image2pipe',  # Pipe as images
            '-vcodec', 'mjpeg',  # Output as MJPEG (JPEG frames)
            '-q:v', '5',  # Quality (lower = better, but balanced)
            '-vf', 'fps=30,scale=640:360',  # Throttle FPS and resize
            'pipe:1'  # Output to stdout
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1)

    def _capture_loop(self):
        last_send_time = time.time()
        backoff = 1  # For exponential backoff on errors
        if 'rtsp' in self.camera_url.lower() and self.process:
            # Read from FFmpeg stdout for RTSP
            buffer = b''
            while self.is_running:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    print(f"FFmpeg ended for {self.name}, restarting after {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8)  # Exponential backoff up to 8s
                    self.process.terminate()
                    self._start_ffmpeg()
                    continue
                buffer += chunk
                while True:
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9', start + 2)
                    if start == -1 or end == -1:
                        break
                    jpg_data = buffer[start:end + 2]
                    buffer = buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.frame = frame
                    time.sleep(0.033)
                # Heartbeat check
                if time.time() - last_send_time > 4:
                    last_send_time = time.time()
                    # Force a send in stream_to_server
        else:
            # OpenCV loop for non-RTSP
            while self.is_running:
                if not self.cap.isOpened():
                    print(f"Reconnecting to {self.name} after {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    self.cap.open(self.camera_url)
                    continue
                
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = cv2.resize(frame, (640, 360))
                    with self.lock:
                        self.frame = frame
                else:
                    print(f"Capture error on {self.name}, retrying...")
                    self.cap.release()
                    time.sleep(0.5)
                
                time.sleep(0.033)
                # Heartbeat check (handled in stream_to_server)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.terminate()

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
            # "CAMERA_1": "1",
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
        frame_counts = {cid: 0 for cid in self.cameras}  # For FPS logging
        last_log_time = time.time()
        last_send_times = {cid: time.time() for cid in self.cameras}  # For heartbeat
        while self.is_running:
            if not self.connected and not self.connect_to_server():
                time.sleep(3)
                continue

            now = time.time()
            for camera_id, cam in self.cameras.items():
                frame = cam.get_frame()
                if frame is None:
                    continue

                # Change detection: Skip if unchanged, but force every 1s
                frame_hash = hashlib.md5(frame.tobytes()).digest()
                force_send = (now - last_send_times[camera_id] > 1)  # Force every 1s
                if not force_send and frame_hash == cam.last_frame_hash:
                    # Heartbeat: If no change but >4s since last send, force a duplicate
                    if now - last_send_times[camera_id] > 4:
                        pass  # Proceed to send duplicate
                    else:
                        continue
                cam.last_frame_hash = frame_hash
                last_send_times[camera_id] = now

                # Encode and send
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                try:
                    self.sio.emit('frame', {
                        'camera_id': camera_id,
                        'frame': buf.tobytes()
                    })
                    frame_counts[camera_id] += 1
                except Exception as e:
                    print(f"Emit error: {e}")
                    self.connected = False
                    break

            # Log FPS every 10s
            if now - last_log_time > 10:
                for cid, count in frame_counts.items():
                    fps = count / 10  # Approximate sent FPS
                    print(f"{datetime.datetime.now()} - Camera {cid}: Sent FPS ~{fps}")
                frame_counts = {cid: 0 for cid in self.cameras}
                last_log_time = now

            time.sleep(0.033)  # ~30 FPS max

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
    import numpy as np  # Needed for imdecode
    main()