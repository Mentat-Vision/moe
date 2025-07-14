# Mixture of Experts Vision System

A centralized vision processing system using a mixture of expert models (YOLO
object detection and BLIP image captioning) with a single WebSocket server
architecture.

## System Architecture

The system uses a **centralized server architecture** where:

- **Single WebSocket Server** handles all client connections on one port
- **Expert Workers** (YOLO, BLIP) process frames asynchronously via internal
  queues
- **Multi-Camera Client** connects to the central server and sends frames to
  specific experts

## Sampo Server (Current Backend)

- **Sampo** is currently an **8x Tesla V100 GPU, 500GB RAM server**
- This may change in the future to a cloud compute platform such as RunPod, AWS,
  or Google Cloud

## Project Structure

```
MOE/
├── mentatClient/           # Client-side code
│   ├── clientMain.py      # Main multi-camera client
│   ├── config.env         # Client configuration
│   ├── requirements.txt   # Client dependencies
│   ├── modelsDownload.py  # Model download utility
│   └── venv/              # Client virtual environment
├── mentatSampo/           # Server-side code
│   ├── serverMain.py      # Central WebSocket server
│   ├── config.env         # Server configuration
│   ├── requirements_server.txt  # Server dependencies
│   ├── experts/           # Expert worker modules
│   │   ├── baseWorker.py  # Base worker class
│   │   ├── serverYolo.py  # YOLO expert worker
│   │   └── serverBlip.py  # BLIP expert worker
│   ├── modelsYolo/        # YOLO model files
│   └── venv/              # Server virtual environment
└── README.md              # This file
```

## Quick Start

### 1. Server Setup (Run on Sampo/Cloud)

```bash
cd mentatSampo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_server.txt

# Edit config.env if needed
python serverMain.py
```

### 2. Client Setup (Run locally)

```bash
cd mentatClient
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Edit config.env to set server IP and cameras
python clientMain.py
```

## Configuration

### Client Configuration (`mentatClient/config.env`)

```bash
# Server connection
SERVER_IP=10.8.162.58      # Sampo server IP
SERVER_PORT=5000           # Central server port

# Camera selection
CAMERAS=0,1                # Use cameras 0 and 1
# CAMERAS=0                # Use only camera 0
# CAMERAS=1,3,5            # Use cameras 1, 3, and 5
```

### Server Configuration (`mentatSampo/config.env`)

```bash
# Server settings
SERVER_PORT=5000

# Model paths
YOLO_MODEL_PATH=modelsYolo/yolo11s.pt
BLIP_MODEL_NAME=Salesforce/blip-image-captioning-base

# GPU settings
USE_GPU=true
CUDA_DEVICE=cuda
```

## How It Works

1. **Central Server**: Single WebSocket server (`serverMain.py`) runs on port
   5000
2. **Expert Workers**: YOLO and BLIP workers process frames asynchronously
3. **Client Protocol**: Client sends JSON messages specifying expert type and
   frame data:
   ```json
   {
   	"expert": "YOLO",
   	"camera_id": 0,
   	"frame": "base64_encoded_image"
   }
   ```
4. **Frame Processing**:
   - YOLO: Object detection every 200ms (5 FPS)
   - BLIP: Image captioning every 3 seconds
5. **Response**: Server returns results directly to client

## Features

- **Multi-Camera Support**: Process multiple cameras simultaneously
- **Expert Routing**: Send frames to specific experts (YOLO or BLIP)
- **Async Processing**: Non-blocking frame processing with queue management
- **Performance Optimization**: Configurable frame intervals for optimal FPS
- **Clean UI**: Non-overlapping text overlays with proper positioning
- **Automatic Reconnection**: Client automatically reconnects on connection loss
- **GPU Acceleration**: Full GPU support for both YOLO and BLIP models

## Display Layout

- **Top-left**: YOLO detection bounding boxes and labels
- **Top-right**: Camera info, connection status, FPS statistics
- **Bottom-left**: BLIP image captions (up to 3 lines)
- **Center**: Live camera feed with detections

## Performance Tuning

The system is optimized for real-time performance:

- **YOLO Processing**: 5 FPS (200ms intervals)
- **BLIP Processing**: Every 3 seconds
- **Frame Size**: 640x480 for processing
- **JPEG Quality**: 85% for efficient transmission

## Model Management

- **YOLO Models**: Stored in `mentatSampo/modelsYolo/`
- **BLIP Models**: Downloaded automatically from Hugging Face
- **Model Download**: Use `mentatClient/modelsDownload.py` for client-side
  testing

## Troubleshooting

### Connection Issues

- Check server IP and port in `mentatClient/config.env`
- Ensure server is running before starting client
- Verify firewall settings allow WebSocket connections

### Performance Issues

- Reduce camera count if FPS is low
- Increase frame intervals in client code
- Check GPU memory usage on server

### Camera Issues

- Verify camera indices exist on your system
- Check camera permissions
- Try different camera indices if some fail

## Development

### Adding New Experts

1. Create new expert worker in `mentatSampo/experts/`
2. Extend `baseWorker.py` class
3. Add expert initialization in `serverMain.py`
4. Update client to send requests to new expert
