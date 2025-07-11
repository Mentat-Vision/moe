import threading
import time
import os
from flask import Flask, Response, render_template
from camera_stream import CameraStreamWeb

# Camera configuration
CAMERAS = [
    {"id": "webcam0", "index": 0},
    # Add more cameras as needed
]

# Initialize camera streams
camera_streams = []
threads = []
for cam_cfg in CAMERAS:
    cam = CameraStreamWeb(cam_cfg)
    camera_streams.append(cam)
    t = threading.Thread(target=cam.run_stream, daemon=True)
    t.start()
    threads.append(t)

# Flask app
app = Flask(__name__, template_folder="templates")

@app.route('/')
def index():
    return render_template('main_page.html', cameras=CAMERAS)

@app.route('/video/<cam_id>')
def video_feed(cam_id):
    # Find the camera by id
    cam = next((c for c in camera_streams if c.id == cam_id), None)
    if not cam:
        return "Camera not found", 404
    def gen():
        while True:
            frame = cam.get_jpeg()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # ~30 FPS
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True) 