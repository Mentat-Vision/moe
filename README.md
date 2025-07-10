# Mixture of Experts Vision System

A new system architecture based on mixture of experts.

## Goal

Build a comprehensive vision system that aggregates multiple specialized models
(captioning, object detection, facial recognition, etc.) under one intelligent
aggregator model.

## Architecture

- **Expert Models**: Specialized models (Captioning, Object Recognition, Facial
  Recognition, etc)
- **Aggregator Model**: Intelligent coordinator that processes and combines
  expert outputs
- **Unified Interface**: Single point of access for user interface

## Current Models

- **BLIP**: Image captioning
- **Yolo**: Object detection

## Future Models

- Object detection
- Facial recognition
- Scene understanding
- Action recognition
- And more...

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd Models
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models:**

   ```bash
   python modelsDownload.py
   ```

4. **Manual model download (if needed):**
   - For Llama model: Download `llama-3.2-1b-instruct-q4_k_m.gguf` from
     [HuggingFace](https://huggingface.co/TheBloke/Llama-3.2-1B-Instruct-GGUF)
   - Place it in `modelsChat/` directory

## Usage

- **Main system:** `python main.py`
- **Chat analysis:** `python chat.py`
- **Standalone YOLO:** `python yolo.py`

## Repo Notes

- models folder contain experimental models & not aggregated by main.py
- Large model files are not tracked in git (see .gitignore)
- Run `python modelsDownload.py` to get required models
