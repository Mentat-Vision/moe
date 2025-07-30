# server.py
from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import logging

from collections import deque
import cv2
import numpy as np
from yolo import initialize_detector, process_frame, get_all_detections, get_stats, get_annotated_frame
from blip import initialize_captioner, process_frame as blip_process_frame, get_all_captions, get_stats as blip_get_stats, switch_model as blip_switch_model, get_current_model_info

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
                
                # Process frame through BLIP captioner (balanced speed)
                if cam_id not in self.last_broadcast_time or (now - self.last_broadcast_time.get(f"{cam_id}_blip", 0)) >= 0.5:
                    try:
                        blip_process_frame(cam_id, jpg_data, now)
                        self.last_broadcast_time[f"{cam_id}_blip"] = now
                    except Exception as e:
                        logger.warning(f"BLIP processing error for {cam_id}: {e}")
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

@app.route("/api/blip/models")
def blip_available_models():
    """Get available BLIP models"""
    return jsonify({
        "models": [
            {
                "id": "blip-base",
                "name": "BLIP Base",
                "model_name": "Salesforce/blip-image-captioning-base",
                "type": "blip",
                "description": "Original BLIP model - fast and lightweight"
            },
            {
                "id": "blip-large", 
                "name": "BLIP Large",
                "model_name": "Salesforce/blip-image-captioning-large",
                "type": "blip",
                "description": "Original BLIP model - better quality but slower"
            },
            {
                "id": "blip2-opt",
                "name": "BLIP-2 OPT-2.7B",
                "model_name": "Salesforce/blip2-opt-2.7b",
                "type": "blip2",
                "description": "BLIP-2 with OPT language model - improved captions"
            },
            {
                "id": "blip2-flan-t5",
                "name": "BLIP-2 Flan-T5-XL",
                "model_name": "Salesforce/blip2-flan-t5-xl",
                "type": "blip2", 
                "description": "BLIP-2 with Flan-T5 - highest quality captions"
            },
            {
                "id": "instructblip-vicuna",
                "name": "InstructBLIP Vicuna-7B",
                "model_name": "Salesforce/instructblip-vicuna-7b",
                "type": "instructblip",
                "description": "InstructBLIP with Vicuna - instruction-following captions"
            },
            {
                "id": "instructblip-flan-t5",
                "name": "InstructBLIP Flan-T5-XL", 
                "model_name": "Salesforce/instructblip-flan-t5-xl",
                "type": "instructblip",
                "description": "InstructBLIP with Flan-T5 - customizable captions"
            }
        ]
    })

@app.route("/api/blip/switch_model", methods=["POST"])
def switch_blip_model():
    """Switch BLIP model"""
    try:
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({"success": False, "error": "Missing model_name parameter"}), 400
        
        model_name = data['model_name']
        result = blip_switch_model(model_name)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error switching BLIP model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

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