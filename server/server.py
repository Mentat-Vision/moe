# server.py
from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

from collections import deque

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class StreamManager:
    def __init__(self):
        self.frames = {}  # Store JPG bytes directly
        self.locks = {}
        self.fps = {}
        self.timestamps = {}
        self.status = {}
        self.clients = {}
        self.names = {}
        self.video_clients = {}  # Track clients requesting video streams
        self.rooms = {}  # New: Rooms for broadcasting
        self.last_broadcast_time = {}  # Track last broadcast time per camera

    def update(self, cam_id, jpg_data, client_id):
        if cam_id not in self.locks:
            self.locks[cam_id] = threading.Lock()
        with self.locks[cam_id]:
            try:
                self.frames[cam_id] = jpg_data  # Store raw JPG bytes
                self.clients[cam_id] = client_id
                now = time.time()
                if cam_id not in self.timestamps:
                    self.timestamps[cam_id] = deque(maxlen=60)  # Increased for smoother FPS
                self.timestamps[cam_id].append(now)
                if len(self.timestamps[cam_id]) > 1:
                    dt = self.timestamps[cam_id][-1] - self.timestamps[cam_id][0]
                    self.fps[cam_id] = round((len(self.timestamps[cam_id]) - 1) / dt, 1) if dt > 0 else 0
                self.status[cam_id] = {'status': 'active', 'last_update': now, 'fps': self.fps.get(cam_id, 0)}
                
                # Throttle video broadcasts to ~30 FPS max
                if cam_id not in self.last_broadcast_time or (now - self.last_broadcast_time[cam_id]) >= 0.033:
                    self.broadcast_video_frame(cam_id, jpg_data)
                    self.last_broadcast_time[cam_id] = now
            except Exception as e:
                print(f"Frame error {cam_id}: {e}")

    def broadcast_video_frame(self, cam_id, jpg_data):
        if cam_id in self.video_clients and self.video_clients[cam_id] and cam_id in self.rooms:
            # Broadcast to the room once (efficient for many clients)
            socketio.emit('video_frame', {
                'camera_id': cam_id,
                'frame_data': jpg_data
            }, room=self.rooms[cam_id])

    def add_video_client(self, cam_id, client_id):
        if cam_id not in self.video_clients:
            self.video_clients[cam_id] = set()
        self.video_clients[cam_id].add(client_id)
        
        # Join the client to the camera's room
        if cam_id not in self.rooms:
            self.rooms[cam_id] = f"room_{cam_id}"
        socketio.server.enter_room(client_id, self.rooms[cam_id])
        print(f"Client {client_id} subscribed to video stream {cam_id}")

    def remove_video_client(self, client_id):
        for cam_id in list(self.video_clients.keys()):
            if client_id in self.video_clients[cam_id]:
                self.video_clients[cam_id].discard(client_id)
                if cam_id in self.rooms:
                    socketio.server.leave_room(client_id, self.rooms[cam_id])

    def get(self, cam_id):
        with self.locks.get(cam_id, threading.Lock()):
            return self.frames.get(cam_id)

    def cleanup(self, client_id):
        self.remove_video_client(client_id)
        for cam_id, cid in list(self.clients.items()):
            if cid == client_id:
                for d in [self.frames, self.locks, self.fps, self.timestamps, self.status, self.clients]:
                    d.pop(cam_id, None)

manager = StreamManager()

@app.route("/")
def index():
    return render_template("dashboard.html")

# Keep MJPEG endpoint as fallback
@app.route("/video/<cam_id>.mjpg")
def stream(cam_id):
    def gen():
        last_frame = None
        while True:
            jpg_bytes = manager.get(cam_id)
            if jpg_bytes is not None and jpg_bytes != last_frame:
                last_frame = jpg_bytes
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            time.sleep(0.033)  # Cap at ~30 FPS
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/cameras")
def cameras():
    now = time.time()
    cams = []
    for cam_id, s in manager.status.items():
        if now - s["last_update"] > 10:  # Increased to 10s
            s["status"] = "inactive"
            s["fps"] = 0
        cams.append({"id": cam_id, "name": manager.names.get(cam_id, cam_id), "status": s["status"], "fps": s["fps"], "last_update": s["last_update"]})
    return jsonify(cams)

@socketio.on("connect")
def connect():
    emit("connected", {"client_id": request.sid})

@socketio.on("disconnect")
def disconnect():
    manager.cleanup(request.sid)

@socketio.on("request_video_stream")
def request_video_stream(data):
    cam_id = data.get("camera_id")
    if cam_id:
        manager.add_video_client(cam_id, request.sid)

@socketio.on("frame")
def frame(data):
    cam_id = data.get("camera_id")
    f = data.get("frame")  # Now binary JPG bytes
    if cam_id and f:
        manager.update(cam_id, f, request.sid)
        emit("frame_received", {"camera_id": cam_id, "status": "success"})
    else:
        emit("frame_received", {"camera_id": cam_id, "status": "error", "message": "Missing data"})

@socketio.on("register_camera")
def register(data):
    cam_id = data.get("camera_id")
    if cam_id:
        manager.names[cam_id] = data.get("name", cam_id)
        emit("camera_registered", {"camera_id": cam_id, "status": "success"})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)