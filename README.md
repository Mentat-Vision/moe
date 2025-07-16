# Mentat - Real-time Video Monitoring System

A distributed video monitoring system that captures video from multiple cameras and displays them in a web dashboard.

## What It Does

- **Local Client**: Captures video from webcams and RTSP cameras
- **Server**: Receives video streams and serves a web dashboard
- **Dashboard**: Shows live video feeds with real-time FPS and status

## Architecture

```
Local Client (WebSocket) → Server (WebSocket) → Dashboard (HTTP MJPEG)
```

- **WebSocket**: Efficient real-time video transmission from client to server
- **HTTP MJPEG**: Standard video streaming for web dashboard compatibility

## Quick Start

### 1. Start Server
```bash
cd server
pip install -r requirements.txt
python server.py
```
Dashboard: http://localhost:5000

### 2. Start Client
```bash
cd local
pip install -r requirements.txt
python local.py
```

## Configuration

### Camera Setup
Edit `local/local.py` to configure cameras:
```python
self.camera_config = {
    "CAMERA_0": "0",  # Webcam 0
    "CAMERA_1": "1",  # Webcam 1
    "CAMERA_2": "2",  # Webcam 2
    # "CAMERA_RTSP_101": "rtsp://user:pass@ip:port/stream",  # RTSP camera
}
```

### Server URL
Set environment variable or pass as argument:
```bash
export MENTAT_SERVER_URL="http://10.8.162.58:5000"
python local.py
# or
python local.py http://10.8.162.58:5000
```

## Features

- **Multi-camera Support**: Webcams and RTSP cameras
- **Real-time FPS**: Live frame rate monitoring
- **Auto-reconnection**: Handles network interruptions
- **Responsive Dashboard**: Works on desktop and mobile
- **Fullscreen Mode**: Click any camera for fullscreen view

## Network Ports

- **Port 5000**: HTTP dashboard + WebSocket server
- **No additional ports needed**

## Troubleshooting

- **No video**: Check camera permissions and indices
- **Connection failed**: Verify server IP and network connectivity
- **Low FPS**: Check network bandwidth and camera performance
