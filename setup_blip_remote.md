# BLIP Remote Processing Setup

This setup allows you to offload BLIP computation to Sampo while keeping the
video preview on your Mac.

## Files Created

1. **`mentatSampo/blip_server.py`** - Server that runs on Sampo
2. **`blip_client.py`** - Client that runs on your Mac
3. **`mentatSampo/requirements_blip.txt`** - Dependencies for the server

## Setup Instructions

### On Sampo (Server):

1. Navigate to the mentatSampo directory:

   ```bash
   cd mentatSampo
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements_blip.txt
   ```

3. Run the BLIP server:
   ```bash
   python blip_server.py
   ```

The server will start on port 5001 and listen for image captioning requests.

### On Your Mac (Client):

1. Update the server URL in `blip_client.py` if needed:

   ```python
   SERVER_URL = "http://10.8.162.58:5001/caption"  # Update IP if different
   ```

2. Install required dependencies:

   ```bash
   pip install opencv-python requests numpy
   ```

3. Run the client:
   ```bash
   python blip_client.py
   ```

## How It Works

1. **Client (Mac)**: Captures video frames and sends them to Sampo every 45
   frames
2. **Server (Sampo)**: Processes images with BLIP model and returns captions
3. **Client (Mac)**: Displays video with caption overlay in real-time

## Features

- ✅ Real-time video preview on Mac
- ✅ Caption overlay with black background
- ✅ Background processing to avoid blocking video
- ✅ Optimized frame encoding (JPEG compression)
- ✅ Error handling and timeout protection
- ✅ Same visual experience as local processing

## Troubleshooting

- **Connection issues**: Check if Sampo IP is correct in `blip_client.py`
- **Performance**: Adjust `caption_interval` in client for different processing
  frequency
- **Quality**: Modify JPEG quality in `_encode_frame()` method
