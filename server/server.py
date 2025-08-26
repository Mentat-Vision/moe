# server.py
from flask import Flask, render_template, request, Response, jsonify, url_for, send_from_directory
from urllib.parse import quote
import os
from flask_socketio import SocketIO, emit
import threading
import time
import logging

from collections import deque
import cv2
import numpy as np
from system1.yolo import initialize_detector, process_frame, get_all_detections, get_stats, get_annotated_frame
from system1.blip import initialize_captioner, process_frame as blip_process_frame, get_all_captions, get_stats as blip_get_stats, get_current_model_info
from system1.face import initialize_face_recognizer, process_frame as face_process_frame, get_all_camera_faces, get_face_stats, get_face_database, update_face_name, delete_face, delete_all_faces, toggle_auto_register, get_face_annotated_frame

logger = logging.getLogger(__name__)

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

    def rotate_esp32_image(self, jpg_data):
        """Rotate ESP32 camera image 180 degrees"""
        try:
            # Decode JPG to numpy array
            frame_array = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Rotate 180 degrees
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Encode back to JPG
                _, rotated_jpg = cv2.imencode('.jpg', rotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return rotated_jpg.tobytes()
            else:
                return jpg_data  # Return original if decode fails
        except Exception as e:
            logger.warning(f"Error rotating ESP32 image: {e}")
            return jpg_data  # Return original if rotation fails

    def update(self, cam_id, jpg_data, client_id):
        if cam_id not in self.locks:
            self.locks[cam_id] = threading.Lock()
        with self.locks[cam_id]:
            try:
                # Rotate ESP32 camera image 180 degrees before processing
                if cam_id == "ESP32_CAMERA":
                    jpg_data = self.rotate_esp32_image(jpg_data)
                
                self.frames[cam_id] = jpg_data  # Store processed JPG bytes
                self.clients[cam_id] = client_id
                now = time.time()
                if cam_id not in self.timestamps:
                    self.timestamps[cam_id] = deque(maxlen=60)  # Increased for smoother FPS
                self.timestamps[cam_id].append(now)
                if len(self.timestamps[cam_id]) > 1:
                    dt = self.timestamps[cam_id][-1] - self.timestamps[cam_id][0]
                    self.fps[cam_id] = round((len(self.timestamps[cam_id]) - 1) / dt, 1) if dt > 0 else 0
                self.status[cam_id] = {'status': 'active', 'last_update': now, 'fps': self.fps.get(cam_id, 0)}
                
                # Always broadcast video frames immediately for max FPS
                self.broadcast_video_frame(cam_id, jpg_data)
                self.last_broadcast_time[cam_id] = now
                
                # Process frame through YOLO detector (non-blocking, less frequently)
                if cam_id not in self.last_broadcast_time or (now - self.last_broadcast_time.get(f"{cam_id}_yolo", 0)) >= 0.5:
                    try:
                        process_frame(cam_id, jpg_data, now)
                        self.last_broadcast_time[f"{cam_id}_yolo"] = now
                    except Exception as e:
                        logger.warning(f"YOLO processing error for {cam_id}: {e}")
                
                # Process frame through BLIP captioner (maximum speed)
                if cam_id not in self.last_broadcast_time or (now - self.last_broadcast_time.get(f"{cam_id}_blip", 0)) >= 0.1:
                    try:
                        blip_process_frame(cam_id, jpg_data, now)
                        self.last_broadcast_time[f"{cam_id}_blip"] = now
                    except Exception as e:
                        logger.warning(f"BLIP processing error for {cam_id}: {e}")
                
                # Process frame through face recognition (every 300ms for good responsiveness)
                if cam_id not in self.last_broadcast_time or (now - self.last_broadcast_time.get(f"{cam_id}_face", 0)) >= 0.3:
                    try:
                        face_process_frame(cam_id, jpg_data, now)
                        self.last_broadcast_time[f"{cam_id}_face"] = now
                    except Exception as e:
                        logger.warning(f"Face processing error for {cam_id}: {e}")
            except Exception as e:
                print(f"Frame error {cam_id}: {e}")

    def broadcast_video_frame(self, cam_id, jpg_data):
        if cam_id in self.video_clients and self.video_clients[cam_id] and cam_id in self.rooms:
            # Always use raw frame for maximum FPS - annotated frames cause lag
            # Only show annotated frames when explicitly requested
            frame_to_send = jpg_data
            
            # Only use annotated frames if YOLO is on AND boxes are enabled AND we have a recent annotated frame
            if yolo_detector.enabled and yolo_detector.draw_boxes:
                annotated_frame = get_annotated_frame(cam_id)
                if annotated_frame:
                    frame_to_send = annotated_frame
            
            # Broadcast to the room once (efficient for many clients)
            socketio.emit('video_frame', {
                'camera_id': cam_id,
                'frame_data': frame_to_send
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

# Initialize YOLO detector
yolo_detector = initialize_detector()
print("YOLO detector initialized with multi-GPU support")

# Initialize BLIP captioner
blip_captioner = initialize_captioner()
print("BLIP captioner initialized with multi-GPU support")

# Initialize Face Recognizer (disabled by default, will be enabled via toggle)
face_recognizer = initialize_face_recognizer(enabled=False)
print("Face recognizer initialized (disabled by default)")

# Register callback to broadcast detection results
def broadcast_detection_result(result_data):
    """Broadcast YOLO detection results to connected clients"""
    socketio.emit('detection_result', result_data)

yolo_detector.register_callback(broadcast_detection_result)

# Register callback to broadcast caption results
def broadcast_caption_result(result_data):
    """Broadcast BLIP caption results to connected clients"""
    socketio.emit('caption_result', result_data)

blip_captioner.register_callback(broadcast_caption_result)

# Register callback to broadcast face recognition results
def broadcast_face_result(result_data):
    """Broadcast face recognition results to connected clients"""
    socketio.emit('face_result', result_data)

face_recognizer.register_callback(broadcast_face_result)

@app.route("/")
def index():
    return render_template("dashboard.html")

# Keep MJPEG endpoint as fallback
@app.route("/video/<cam_id>.mjpg")
def stream(cam_id):
    def gen():
        last_frame = None
        while True:
            # Try to get annotated frame first, fallback to original
            annotated_frame = get_annotated_frame(cam_id)
            jpg_bytes = annotated_frame if annotated_frame else manager.get(cam_id)
            
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

@app.route("/api/detections")
def all_detections():
    """Get latest detection results for all cameras"""
    return jsonify(get_all_detections())

@app.route("/api/detections/<cam_id>")
def camera_detections(cam_id):
    """Get latest detection results for a specific camera"""
    from yolo import get_detections
    return jsonify(get_detections(cam_id))

@app.route("/api/yolo/stats")
def yolo_stats():
    """Get YOLO processing statistics"""
    return jsonify(get_stats())

@app.route("/api/yolo/toggle_boxes", methods=["POST"])
def toggle_bounding_boxes():
    """Toggle bounding box drawing on/off"""
    global yolo_detector
    current_state = yolo_detector.draw_boxes
    yolo_detector.draw_boxes = not current_state
    return jsonify({
        "draw_boxes": yolo_detector.draw_boxes,
        "message": f"Bounding boxes {'enabled' if yolo_detector.draw_boxes else 'disabled'}"
    })

@app.route("/api/yolo/toggle", methods=["POST"])
def toggle_yolo():
    """Toggle YOLO processing on/off"""
    global yolo_detector
    current_state = yolo_detector.enabled
    yolo_detector.enabled = not current_state
    
    # Clear annotated frames when disabling to force fallback to raw frames
    if not yolo_detector.enabled:
        yolo_detector.annotated_frames.clear()
    
    return jsonify({
        "enabled": yolo_detector.enabled,
        "message": f"YOLO processing {'enabled' if yolo_detector.enabled else 'disabled'}"
    })

@app.route("/api/yolo/status")
def yolo_status():
    """Get current YOLO status"""
    global yolo_detector
    return jsonify({
        "enabled": yolo_detector.enabled,
        "draw_boxes": yolo_detector.draw_boxes,
        "detection_interval": yolo_detector.detection_interval,
        "gpu_count": len(yolo_detector.models)
    })

# BLIP API endpoints
@app.route("/api/blip/stats")
def blip_stats():
    """Get BLIP processing statistics"""
    return jsonify(blip_get_stats())

@app.route("/api/blip/status")
def blip_status():
    """Get current BLIP status and model info"""
    return jsonify(get_current_model_info())

@app.route("/api/blip/toggle", methods=["POST"])
def toggle_blip():
    """Toggle BLIP processing on/off"""
    global blip_captioner
    current_state = blip_captioner.enabled
    blip_captioner.enabled = not current_state
    
    return jsonify({
        "enabled": blip_captioner.enabled,
        "message": f"BLIP processing {'enabled' if blip_captioner.enabled else 'disabled'}"
    })

# Face Recognition API endpoints
@app.route("/api/face/stats")
def face_stats():
    """Get face recognition statistics"""
    return jsonify(get_face_stats())

@app.route("/api/face/database")
def face_database():
    """Get all registered faces"""
    return jsonify(get_face_database())

@app.route("/api/face/faces")
def camera_faces():
    """Get recent face results for all cameras"""
    return jsonify(get_all_camera_faces())

@app.route("/api/face/faces/<cam_id>")
def camera_faces_by_id(cam_id):
    """Get recent face results for a specific camera"""
    from system1.face import get_camera_faces
    return jsonify(get_camera_faces(cam_id))

@app.route("/api/demo/faces")
def demo_faces():
    """List demo face images from likely faces folders.

    Checks in order:
    - server/static/faces
    - server/faces
    - <repo_root>/faces
    """
    results = []

    # 1) server/static/faces (served via media route)
    static_faces_dir = os.path.join(app.static_folder, "faces")
    if os.path.isdir(static_faces_dir):
        for fname in sorted(os.listdir(static_faces_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                results.append({
                    "name": os.path.splitext(fname)[0],
                    "url": f"/media/faces/static/{quote(fname)}"
                })

    # 2) server/faces (serve via a static URL prefix mapping under /static/..)
    server_dir = os.path.dirname(os.path.abspath(__file__))
    server_faces_dir = os.path.join(server_dir, "faces")
    if os.path.isdir(server_faces_dir):
        for fname in sorted(os.listdir(server_faces_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                results.append({
                    "name": os.path.splitext(fname)[0],
                    "url": f"/media/faces/server/{quote(fname)}"
                })

    # 3) repo root /faces
    repo_root = os.path.abspath(os.path.join(server_dir, os.pardir))
    root_faces_dir = os.path.join(repo_root, "faces")
    if os.path.isdir(root_faces_dir):
        for fname in sorted(os.listdir(root_faces_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                results.append({
                    "name": os.path.splitext(fname)[0],
                    "url": f"/media/faces/root/{quote(fname)}"
                })

    return jsonify(results)

@app.route('/media/faces/<source>/<path:filename>')
def serve_faces_media(source, filename):
    """Serve face images from known directories via a stable media route."""
    server_dir = os.path.dirname(os.path.abspath(__file__))
    static_faces_dir = os.path.join(app.static_folder, "faces")
    server_faces_dir = os.path.join(server_dir, "faces")
    repo_root = os.path.abspath(os.path.join(server_dir, os.pardir))
    root_faces_dir = os.path.join(repo_root, "faces")

    source_map = {
        'static': static_faces_dir,
        'server': server_faces_dir,
        'root': root_faces_dir,
    }
    base_dir = source_map.get(source)
    if base_dir and os.path.isdir(base_dir):
        # Security: ensure resolved path is within base_dir
        requested_path = os.path.normpath(os.path.join(base_dir, filename))
        if requested_path.startswith(os.path.abspath(base_dir)) and os.path.exists(requested_path):
            return send_from_directory(base_dir, filename)
    return Response(status=404)

@app.route('/media/chat/<path:name>')
def serve_chat_media(name):
    """Serve chat image attachments from server/chat.
    If extension is omitted, try common image extensions.
    """
    server_dir = os.path.dirname(os.path.abspath(__file__))
    chat_dir = os.path.join(server_dir, 'chat')
    if not os.path.isdir(chat_dir):
        return Response(status=404)
    allowed_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    base, ext = os.path.splitext(name)
    candidates = []
    if ext:
        candidates = [name]
    else:
        candidates = [base + e for e in allowed_exts]
    for candidate in candidates:
        path = os.path.join(chat_dir, candidate)
        if os.path.exists(path):
            return send_from_directory(chat_dir, candidate)
    return Response(status=404)

@app.route("/api/face/toggle", methods=["POST"])
def toggle_face_recognition():
    """Toggle face recognition on/off"""
    global face_recognizer
    current_state = face_recognizer.enabled
    face_recognizer.enabled = not current_state
    
    return jsonify({
        "enabled": face_recognizer.enabled,
        "message": f"Face recognition {'enabled' if face_recognizer.enabled else 'disabled'}"
    })

@app.route("/api/face/status")
def face_status():
    """Get current face recognition status"""
    global face_recognizer
    return jsonify({
        "enabled": face_recognizer.enabled,
        "auto_register": face_recognizer.auto_register,
        "use_embeddings": face_recognizer.use_embeddings,
        "total_faces": len(face_recognizer.faces_db),
        "model_loaded": face_recognizer.arcface_model.model_loaded if face_recognizer.arcface_model else False
    })

@app.route("/api/face/auto_register/toggle", methods=["POST"])
def toggle_auto_register():
    """Toggle auto-registration of new faces"""
    result = toggle_auto_register()
    return jsonify({
        "auto_register": result,
        "message": f"Auto-registration {'enabled' if result else 'disabled'}"
    })

@app.route("/api/face/update_name", methods=["POST"])
def update_face_name_endpoint():
    """Update a face name"""
    data = request.get_json()
    face_id = data.get('face_id')
    new_name = data.get('new_name')
    
    if not face_id or not new_name:
        return jsonify({"success": False, "error": "Missing face_id or new_name"}), 400
    
    success = update_face_name(face_id, new_name)
    return jsonify({"success": success})

@app.route("/api/face/delete", methods=["POST"])
def delete_face_endpoint():
    """Delete a face"""
    data = request.get_json()
    face_id = data.get('face_id')
    
    if not face_id:
        return jsonify({"success": False, "error": "Missing face_id"}), 400
    
    success = delete_face(face_id)
    return jsonify({"success": success})

@app.route("/api/face/delete_all", methods=["POST"])
def delete_all_faces_endpoint():
    """Delete all faces"""
    success = delete_all_faces()
    return jsonify({"success": success})


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
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)