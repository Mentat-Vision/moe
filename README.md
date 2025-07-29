# Mentat - Multi-Camera AI Surveillance System

A high-performance real-time surveillance system with YOLOv11m object detection, supporting multiple camera types and optimized for GPU acceleration.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Client  │───▶│   Mentat Server  │───▶│  Web Dashboard  │
│                 │    │                  │    │                 │
│ • Camera feeds  │    │ • Frame routing  │    │ • Live streams  │
│ • Frame capture │    │ • YOLO processing│    │ • Detections    │
│ • Preprocessing │    │ • WebSocket hub  │    │ • Controls      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cameras       │    │   GPU Cluster    │    │   Browser UI    │
│                 │    │                  │    │                 │
│ • Local webcams │    │ • 8x Tesla V100  │    │ • Real-time     │
│ • RTSP streams  │    │ • Load balancing │    │ • Multi-camera  │
│ • ESP32 cameras │    │ • YOLO inference │    │ • Bounding boxes│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
cd server
python server.py
```
Server runs on: **http://localhost:5001**

### 3. Connect Cameras
```bash
cd local
python local.py
```
Or specify custom server: `python local.py http://10.8.120.100:5001`

### 4. Access Web Dashboard
Open browser to: **http://localhost:5001**

## Supported Camera Types

### 📹 Local Webcams
```python
"CAMERA_0": "0",        # /dev/video0
"CAMERA_1": "1",        # /dev/video1
```

### 📡 RTSP Streams
```python
"CAMERA_RTSP": "rtsp://user:pass@10.19.55.20:554/Streaming/Channels/101"
```

### 🎥 ESP32 Cameras
```python
"ESP32_CAMERA": "http://10.8.120.220/stream"
```

Edit camera sources in `local/local.py` → `camera_config`

## Features

### 🚀 High Performance
- **Optimized for speed**: 20-30+ FPS video streaming
- **GPU acceleration**: 8x Tesla V100 load balancing  
- **Non-blocking processing**: Video streams independently of AI
- **Low latency**: <100ms frame-to-display pipeline

### 🤖 AI Object Detection
- **YOLOv11m model**: State-of-the-art accuracy
- **Real-time bounding boxes**: Live object detection overlay
- **80+ object classes**: People, vehicles, animals, objects
- **Confidence filtering**: Adjustable detection thresholds

### 🎛️ Interactive Controls
- **YOLO Toggle**: Enable/disable AI processing
- **Bounding Box Toggle**: Show/hide detection overlays
- **Multi-camera Grid**: Scalable camera layout
- **Fullscreen Mode**: Click any camera for full view

### 📊 Monitoring & Stats
- **Live FPS counters**: Per-camera performance metrics
- **GPU utilization**: Real-time processing statistics  
- **Detection counts**: Object detection analytics
- **Connection status**: Camera health monitoring

## Performance Optimizations

### 🔧 Video Pipeline
- **Frame resolution**: Optimized 480x270 for speed
- **JPEG compression**: Quality 50 for fast encoding
- **Time-based throttling**: Eliminates expensive frame diffing
- **Minimal buffering**: Reduced latency throughout system

### ⚡ YOLO Processing  
- **GPU load balancing**: Distributes cameras across 8 GPUs
- **Async processing**: YOLO runs independently (0.5s intervals)
- **Smart queuing**: Small buffers prevent processing lag
- **Optimized encoding**: Lower quality for annotated frames

### 🌐 Network Layer
- **WebSocket streaming**: Efficient real-time communication
- **Room-based broadcasting**: Scales to multiple viewers
- **Compressed frames**: Optimized data transmission
- **Connection pooling**: Robust reconnection handling

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Tesla V100 recommended)
- **RAM**: 16GB+ (more for multiple cameras)
- **CPU**: Multi-core processor for camera handling
- **Network**: Gigabit ethernet for multiple HD streams

### Software
- **Python**: 3.8+
- **CUDA**: 11.8+ 
- **FFmpeg**: For RTSP stream processing
- **Modern Browser**: Chrome/Firefox for dashboard

## Configuration

### Camera Setup (`local/local.py`)
```python
self.camera_config = {
    "CAMERA_0": "0",                    # Local webcam
    "ESP32_CAMERA": "http://10.8.120.220/stream",  # ESP32
    "RTSP_CAM": "rtsp://user:pass@ip:554/path",     # RTSP
}
```

### Server Settings (`server/server.py`)
- **Port**: 5001 (Flask + SocketIO)
- **CORS**: Enabled for cross-origin access
- **Max connections**: Unlimited concurrent clients

### YOLO Configuration (`server/yolo.py`)
- **Model**: YOLOv11m.pt (auto-downloaded)
- **Confidence**: 0.5 threshold
- **Detection interval**: 0.5 seconds
- **GPU assignment**: Round-robin across available GPUs

## API Endpoints

### REST API
- `GET /api/cameras` - List all connected cameras
- `GET /api/detections` - Latest detection results (all cameras)
- `GET /api/detections/<cam_id>` - Specific camera detections
- `GET /api/yolo/stats` - YOLO processing statistics
- `POST /api/yolo/toggle` - Enable/disable YOLO processing
- `POST /api/yolo/toggle_boxes` - Show/hide bounding boxes

### WebSocket Events
- `frame` - Send camera frame to server
- `register_camera` - Register new camera feed
- `video_frame` - Receive video frame (client)
- `detection_result` - Receive YOLO detections (client)

## Troubleshooting

### Common Issues

**Low FPS Performance:**
- Check GPU utilization: `nvidia-smi`
- Disable YOLO temporarily to isolate issue
- Reduce number of concurrent cameras
- Verify network bandwidth

**Camera Connection Failures:**
- Check camera URLs and credentials
- Test RTSP streams with VLC player
- Verify firewall/network access
- Check camera power and connectivity

**YOLO Processing Errors:**
- Ensure CUDA is properly installed
- Check GPU memory availability
- Verify model download completed
- Review server logs for errors

### Performance Tuning

**For Maximum FPS:**
```python
# In local.py - reduce quality/resolution
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
frame = cv2.resize(frame, (320, 180))
```

**For Better Quality:**
```python
# In local.py - increase quality/resolution  
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
frame = cv2.resize(frame, (640, 360))
```

**YOLO Frequency:**
```python
# In yolo.py - adjust detection interval
self.detection_interval = 1.0  # Slower but less GPU usage
self.detection_interval = 0.2  # Faster but more GPU usage
```

## Development

### Project Structure
```
├── server/              # Main server application
│   ├── server.py       # Flask + SocketIO server
│   ├── yolo.py         # YOLO detection engine
│   ├── static/         # Web assets (CSS, JS)
│   └── templates/      # HTML templates
├── local/              # Camera client
│   └── local.py        # Camera capture + streaming
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Adding New Camera Types
1. Edit `local/local.py` → `CameraStream.start()`
2. Add detection logic for new URL format
3. Implement capture method in `_capture_loop()`
4. Test with single camera first

### Customizing YOLO Models
1. Download model: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
2. Update `yolo.py` → `initialize_detector("model_name.pt")`
3. Adjust confidence threshold as needed
4. Consider model size vs accuracy trade-offs

## License & Credits

- **YOLO**: Ultralytics YOLOv11 implementation
- **Frontend**: Custom HTML5/CSS3/JavaScript
- **Backend**: Flask + SocketIO + OpenCV
- **ESP32**: Custom firmware for camera integration

Built for high-performance multi-camera surveillance with AI object detection.
