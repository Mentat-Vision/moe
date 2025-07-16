# CCTV Dashboard System

A real-time CCTV monitoring system with a local client that captures video streams and a central server that displays them in a web dashboard.

## Architecture

- **Local Client** (`local/`): Captures video from webcams and RTSP cameras, streams to server
- **Server** (`server/`): Flask web server that receives streams and serves dashboard
- **Communication**: MJPEG over HTTP on port 5000

## Setup

### Prerequisites

- Python 3.8+
- Webcams or RTSP cameras
- Network connectivity between local and server machines

### Local Client Setup

1. Navigate to the local directory:
   ```bash
   cd local
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure cameras in `local.py`:
   ```python
   self.camera_config = {
       "CAMERA_0": "0",  # Webcam 0
       "CAMERA_1": "1",  # Webcam 1
       "CAMERA_2": "2",  # Webcam 2
       # Uncomment RTSP cameras as needed:
       # "CAMERA_RTSP_101": "rtsp://user:pass@ip:port/stream",
   }
   ```

4. Run the local client:
   ```bash
   python local.py
   ```

### Server Setup

1. Navigate to the server directory:
   ```bash
   cd server
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python server.py
   ```

4. Open your browser and go to: `http://localhost:5000`

## Features

### Dashboard
- Real-time video streams from all cameras
- Camera status indicators (active/inactive)
- Responsive grid layout
- Fullscreen mode (click on any camera)
- Automatic reconnection

### Camera Support
- **Webcams**: Direct device access (0, 1, 2, etc.)
- **RTSP Cameras**: Network cameras with authentication
- **Status Monitoring**: Automatic detection of camera availability

### Network Configuration
- **Port 5000**: HTTP server and MJPEG streaming
- **Cross-platform**: Works on Windows, Linux, macOS
- **Network Access**: Server runs on `0.0.0.0:5000` for remote access

## Configuration

### Camera Configuration

Edit the `camera_config` dictionary in `local/local.py`:

```python
self.camera_config = {
    # Webcams
    "CAMERA_0": "0",
    "CAMERA_1": "1", 
    "CAMERA_2": "2",
    
    # RTSP Cameras (uncomment and configure)
    # "CAMERA_RTSP_101": "rtsp://username:password@192.168.1.100:554/stream1",
    # "CAMERA_RTSP_102": "rtsp://username:password@192.168.1.101:554/stream1",
}
```

### Server Configuration

Edit `server/server.py` to change:
- Port number (default: 5000)
- Host binding (default: 0.0.0.0)
- Frame rate (default: 10 FPS)
- JPEG quality (default: 80%)

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Verify camera index (0, 1, 2, etc.)
   - Test with `cv2.VideoCapture()` directly

2. **RTSP connection fails**:
   - Verify network connectivity
   - Check credentials and URL format
   - Test with VLC or other RTSP client

3. **Dashboard not loading**:
   - Check server is running on correct port
   - Verify firewall settings
   - Check browser console for errors

4. **Poor video quality**:
   - Adjust JPEG quality in server.py
   - Check network bandwidth
   - Reduce frame rate if needed

### Debug Mode

Run server with debug enabled:
```bash
python server.py
```

Check local client logs for camera connection issues.

## Development

### Adding New Features

1. **New Camera Types**: Extend `CameraStream` class in `local.py`
2. **Dashboard Features**: Modify `dashboard.html` and `dashboard.js`
3. **Server APIs**: Add new routes in `server.py`

### Testing

- Test individual cameras before running full system
- Use dummy video files for development
- Monitor system resources (CPU, memory, network)

## Security Notes

- Change default credentials for RTSP cameras
- Use HTTPS in production
- Implement authentication for dashboard access
- Restrict network access to trusted IPs

## License

This project is for educational and development purposes.
