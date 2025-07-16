from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import threading
import time
import io
from PIL import Image
import base64
from collections import deque

app = Flask(__name__)

class StreamManager:
    def __init__(self):
        self.camera_frames = {}
        self.camera_locks = {}
        self.camera_status = {}
        self.camera_fps = {}
        self.frame_timestamps = {}
    
    def update_frame(self, camera_id: str, frame_data: bytes):
        """Update frame for a specific camera"""
        if camera_id not in self.camera_locks:
            self.camera_locks[camera_id] = threading.Lock()
        
        with self.camera_locks[camera_id]:
            try:
                # Convert bytes to numpy array
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.camera_frames[camera_id] = frame
                
                # Update FPS calculation
                current_time = time.time()
                if camera_id not in self.frame_timestamps:
                    self.frame_timestamps[camera_id] = deque(maxlen=30)  # Keep last 30 timestamps
                
                self.frame_timestamps[camera_id].append(current_time)
                
                # Calculate FPS based on last 30 frames
                if len(self.frame_timestamps[camera_id]) > 1:
                    time_diff = self.frame_timestamps[camera_id][-1] - self.frame_timestamps[camera_id][0]
                    if time_diff > 0:
                        fps = (len(self.frame_timestamps[camera_id]) - 1) / time_diff
                        self.camera_fps[camera_id] = round(fps, 1)
                
                self.camera_status[camera_id] = {
                    'last_update': current_time,
                    'status': 'active',
                    'fps': self.camera_fps.get(camera_id, 0)
                }
            except Exception as e:
                print(f"Error processing frame for {camera_id}: {e}")
    
    def get_frame(self, camera_id: str):
        """Get the latest frame for a camera"""
        if camera_id in self.camera_locks:
            with self.camera_locks[camera_id]:
                return self.camera_frames.get(camera_id)
        return None
    
    def get_camera_status(self):
        """Get status of all cameras"""
        current_time = time.time()
        status = {}
        
        for camera_id in self.camera_frames.keys():
            if camera_id in self.camera_status:
                last_update = self.camera_status[camera_id]['last_update']
                # Mark as inactive if no update in last 5 seconds
                if current_time - last_update > 5:
                    self.camera_status[camera_id]['status'] = 'inactive'
                    self.camera_status[camera_id]['fps'] = 0
                status[camera_id] = self.camera_status[camera_id]
        
        return status

# Global stream manager
stream_manager = StreamManager()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/stream/<camera_id>', methods=['POST'])
def receive_stream(camera_id):
    """Receive MJPEG stream from local client"""
    print(f"Received POST request from {request.remote_addr} for camera {camera_id}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Content-Type: {request.content_type}")
    print(f"Content-Length: {request.content_length}")
    
    try:
        frame_data = request.data
        print(f"Received {len(frame_data)} bytes of frame data")
        stream_manager.update_frame(camera_id, frame_data)
        print(f"Successfully processed frame for {camera_id}")
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print(f"Error receiving stream from {camera_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Generate MJPEG stream for a specific camera"""
    def generate():
        while True:
            frame = stream_manager.get_frame(camera_id)
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send a placeholder frame if no camera data
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Camera {camera_id} - No Signal", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # 10 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras')
def get_cameras():
    """Get list of active cameras"""
    status = stream_manager.get_camera_status()
    cameras = []
    
    for camera_id, camera_status in status.items():
        cameras.append({
            'id': camera_id,
            'name': camera_id.replace('CAMERA_', 'Camera '),
            'status': camera_status['status'],
            'last_update': camera_status['last_update'],
            'fps': camera_status.get('fps', 0)
        })
    
    return jsonify(cameras)

@app.route('/api/camera/<camera_id>/frame')
def get_camera_frame(camera_id):
    """Get a single frame from a camera as base64"""
    frame = stream_manager.get_frame(camera_id)
    if frame is not None:
        # Encode frame as base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'camera_id': camera_id,
            'frame': frame_base64,
            'timestamp': time.time()
        })
    else:
        return jsonify({'error': 'Camera not available'}), 404

if __name__ == '__main__':
    print("Starting CCTV Server...")
    print("Dashboard available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 