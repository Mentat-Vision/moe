# Mixture of Experts Vision System

A new system architecture based on mixture of experts.

## Sampo Server (Current Backend)

- **Sampo** is currently an **8x Tesla V100 GPU, 500GB RAM server** (used by
  deo). This may change in the future to a cloud compute platform such as
  RunPod, AWS, or Google Cloud.

## Project Structure

- `mentatClient/` — All client-side code and dependencies
  - `clientMain.py` — Main client program (runs all experts in parallel)
  - `experts/` — Individual expert clients (e.g. BLIP, YOLO, Llama)
  - `venv/` — Python virtual environment for client dependencies
  - `config.env` — Configuration for ports, cameras, etc.
  - `modelsDownload.py` — Downloads only the models actually used
  - `requirements.txt` — Only the dependencies actually used
  - `modelsYolo/` — Client-side YOLO models (for local testing)
- `mentatSampo/` — All server-side code (run on Sampo or future cloud server)
  - `serverMain.py` — Main server program (runs all experts in parallel)
  - `experts/` — Individual expert servers (e.g. BLIP, YOLO, Llama)
  - `config.env` — Server configuration (ports, models, GPU settings)
  - `modelsYolo/` — Server-side YOLO models (for production inference)
- `mentatBoxhost/` — (Empty, ignored by git, reserved for future use)

## Usage

- **Client/Server split:**

  - Run files in `mentatSampo/` on the server (Sampo or cloud)
  - Run files in `mentatClient/` on your local/client machine

- **Main system:**

  - On the client: `python clientMain.py` (runs all expert models in parallel)
  - On the server: `python serverMain.py`

- **Individual expert models:**

  - On the client: `python experts/clientYolo.py` or
    `python experts/clientBlip.py` (from within `mentatClient/`)
  - On the server: `python experts/serverYolo.py` or
    `python experts/serverBlip.py` (from within `mentatSampo/`)

- **Configuration:**
  - Edit `mentatClient/config.env` to set ports, server IP, and which cameras
    are enabled
  - Edit `mentatSampo/config.env` to set server ports, model paths, and GPU
    settings

## Camera Configuration

The client configuration supports flexible camera selection:

```bash
# In mentatClient/config.env
CAMERAS=0,1          # Use cameras 0 and 1
CAMERAS=0             # Use only camera 0
CAMERAS=1,3,5         # Use cameras 1, 3, and 5
CAMERAS=0,2,4         # Use cameras 0, 2, and 4
```

## Model Management

- **Client models**: Stored in `mentatClient/modelsYolo/` for local testing and
  development
- **Server models**: Stored in `mentatSampo/modelsYolo/` for production
  inference
- Each side manages its own models independently
- Use `mentatClient/modelsDownload.py` to download models for client-side
  testing

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd MOE
   ```

2. **Set up the client environment:**

   ```bash
   cd mentatClient
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download required models:**

   ```bash
   python modelsDownload.py
   ```

4. **Edit configuration files as needed:**
   - `mentatClient/config.env` for client settings (ports, server IP, camera
     selection)
   - `mentatSampo/config.env` for server settings (ports, models, GPU settings)

## Notes

- Large model files and all sensitive/large data are ignored by git (see
  `.gitignore`).
- Only models and dependencies actually used are included in `modelsDownload.py`
  and `requirements.txt`.
- The system is designed to be modular: you can run the main client/server for
  all experts, or run individual expert client/server pairs as needed.
- Camera indices that don't exist will be automatically disabled with a warning
  message.
- Client and server have separate model folders to maintain independence.
