import cv2
import numpy as np
import torch
import threading
import time
import json
from collections import deque, defaultdict
from datetime import datetime

# Option 1: Original BLIP (fastest - current)
from transformers import BlipProcessor, BlipForConditionalGeneration

# Option 2: BLIP-2 (not used for speed optimization)
# from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Option 3: InstructBLIP (not used for speed optimization)  
# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import queue
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BLIPCaptioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", enabled: bool = True):
        """
        Initialize BLIP captioner with multi-GPU support
        
        Args:
            enabled: Whether BLIP processing is enabled
        """
        # Force fastest model and settings
        self.model_name = "Salesforce/blip-image-captioning-base"  # Always use fastest
        self.caption_interval = 0.05  # Caption every 50ms for maximum speed
        self.enabled = enabled
        self.max_image_size = (64, 64)  # Very small for maximum speed
        self.model_type = 'blip'  # Always fastest BLIP
        
        # GPU configuration - distribute across available GPUs
        self.available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
        logger.info(f"Available GPUs for BLIP: {self.available_gpus}")
        
        # Load models on different GPUs for load balancing
        self.models = {}
        self.processors = {}
        
        for i, gpu_id in enumerate(self.available_gpus[:1]):  # Use only 1 GPU for BLIP to avoid competition
            if gpu_id is not None:
                device = f"cuda:{gpu_id}"
                try:
                    processor, model = self._load_model_and_processor(self.model_name, device)
                    
                    self.processors[gpu_id] = processor
                    self.models[gpu_id] = model
                    logger.info(f"Loaded {self.model_type.upper()} model on GPU {gpu_id}")
                except Exception as e:
                    logger.error(f"Failed to load {self.model_type.upper()} model on GPU {gpu_id}: {e}")
                    # Fallback to CPU
                    if not self.models:
                        self._load_cpu_model()
                    break
            else:
                # CPU fallback
                self._load_cpu_model()
                break
        
        if not self.models:
            logger.error("Failed to load BLIP model on any device")
            raise RuntimeError("Could not initialize BLIP model")
        
        # Camera-specific data structures - optimized for speed
        self.camera_queues = defaultdict(lambda: queue.Queue(maxsize=1))  # Single frame queue
        self.camera_threads = {}
        self.camera_gpu_mapping = {}
        self.caption_results = defaultdict(lambda: deque(maxlen=10))  # Store recent captions
        self.camera_fps = defaultdict(float)
        self.camera_last_caption = defaultdict(float)
        self.frame_buffers = defaultdict(lambda: deque(maxlen=30))
        
        # Thread management
        self.running = True
        self.result_callbacks = []
        
        # Processing flags to prevent concurrent processing per camera
        self.processing_flags = defaultdict(bool)
    
    def _load_model_and_processor(self, model_name: str, device: str):
        """Load fastest BLIP model and processor"""
        # Always use original BLIP for maximum speed
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        
        model.to(device)
        model.eval()  # Set to evaluation mode for faster inference
        if device != "cpu":
            model.half()  # Use half precision for speed on GPU
            
        return processor, model
        
    def _load_cpu_model(self):
        """Load BLIP model on CPU as fallback"""
        try:
            processor, model = self._load_model_and_processor(self.model_name, "cpu")
            
            self.processors[0] = processor
            self.models[0] = model
            logger.info(f"Loaded {self.model_type.upper()} model on CPU")
        except Exception as e:
            logger.error(f"Failed to load {self.model_type.upper()} model on CPU: {e}")
            raise
    
    def assign_gpu_to_camera(self, camera_id: str) -> int:
        """Assign a GPU to a camera for load balancing"""
        if camera_id not in self.camera_gpu_mapping:
            # Round-robin GPU assignment
            gpu_id = len(self.camera_gpu_mapping) % len(self.models)
            available_gpu_ids = list(self.models.keys())
            self.camera_gpu_mapping[camera_id] = available_gpu_ids[gpu_id]
            logger.info(f"Assigned camera {camera_id} to BLIP GPU {self.camera_gpu_mapping[camera_id]}")
        return self.camera_gpu_mapping[camera_id]
    
    def generate_caption(self, image: Image.Image, processor, model, device) -> str:
        """
        Generate caption for an image using BLIP/BLIP-2/InstructBLIP
        
        Args:
            image: PIL Image
            processor: BLIP processor
            model: BLIP model
            device: Device string (cuda:0, cpu, etc.)
            
        Returns:
            Generated caption string
        """
        try:
            # Fastest processing - original BLIP style
            inputs = processor(image, return_tensors="pt")
            
            # Move inputs to correct device with maximum speed precision
            if device != "cpu":
                inputs = {k: v.to(device, dtype=torch.float16) for k, v in inputs.items()}
            
            # Optimized for maximum speed - shortest possible captions
            max_len, min_len, beams = 12, 3, 1
            
            with torch.no_grad():
                if device != "cpu":
                    with torch.cuda.amp.autocast():  # Mixed precision for speed
                        generated_ids = model.generate(
                            **inputs,
                            max_length=max_len,
                            min_length=min_len,
                            num_beams=beams,
                            do_sample=False,
                            early_stopping=True,
                            use_cache=True
                        )
                else:
                    generated_ids = model.generate(
                        **inputs,
                        max_length=max_len,
                        min_length=min_len,
                        num_beams=beams,
                        do_sample=False,
                        early_stopping=True,
                        use_cache=True
                    )
            
            # Decode caption
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating caption with {self.model_type.upper()}: {e}")
            return "Caption generation failed"
    
    def add_frame(self, camera_id: str, frame_data: bytes, timestamp: Optional[float] = None):
        """
        Add frame to captioning queue
        
        Args:
            camera_id: Unique camera identifier
            frame_data: JPEG frame data as bytes
            timestamp: Frame timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Update frame buffer for FPS calculation
        self.frame_buffers[camera_id].append(timestamp)
        
        # Calculate FPS
        if len(self.frame_buffers[camera_id]) > 1:
            time_span = self.frame_buffers[camera_id][-1] - self.frame_buffers[camera_id][0]
            if time_span > 0:
                self.camera_fps[camera_id] = (len(self.frame_buffers[camera_id]) - 1) / time_span
        
        # Skip BLIP processing if disabled
        if not self.enabled:
            return
        
        # Only process if enough time has passed (removed processing flag check for speed)
        last_caption = self.camera_last_caption[camera_id]
        if timestamp - last_caption >= self.caption_interval:
            try:
                # Replace any existing frame in queue (we only want the latest)
                while not self.camera_queues[camera_id].empty():
                    try:
                        self.camera_queues[camera_id].get_nowait()
                    except queue.Empty:
                        break
                
                self.camera_queues[camera_id].put_nowait({
                    'frame_data': frame_data,
                    'timestamp': timestamp,
                    'camera_id': camera_id
                })
                self.camera_last_caption[camera_id] = timestamp
                
                # Start processing thread if not already running
                if camera_id not in self.camera_threads or not self.camera_threads[camera_id].is_alive():
                    self.camera_threads[camera_id] = threading.Thread(
                        target=self._process_camera_queue,
                        args=(camera_id,),
                        daemon=True
                    )
                    self.camera_threads[camera_id].start()
                    
            except queue.Full:
                logger.warning(f"Caption queue full for camera {camera_id}, dropping frame")
    
    def _process_camera_queue(self, camera_id: str):
        """Process frames for captioning for a specific camera"""
        gpu_id = self.assign_gpu_to_camera(camera_id)
        model = self.models[gpu_id]
        processor = self.processors[gpu_id]
        device = f"cuda:{gpu_id}" if gpu_id != 0 or torch.cuda.is_available() else "cpu"
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame_info = self.camera_queues[camera_id].get(timeout=1.0)
                
                # Decode JPEG frame
                frame_array = np.frombuffer(frame_info['frame_data'], dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning(f"Failed to decode frame for camera {camera_id}")
                    continue
                
                # Convert BGR to RGB and create PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize image for faster processing with fastest method
                pil_image = pil_image.resize(self.max_image_size, Image.BILINEAR)
                
                # Generate caption
                caption = self.generate_caption(pil_image, processor, model, device)
                
                # Prepare result
                result_data = {
                    'camera_name': camera_id,
                    'caption': caption,
                    'timestamp': datetime.fromtimestamp(frame_info['timestamp']).isoformat(),
                    'unix_timestamp': frame_info['timestamp'],
                    'fps': round(self.camera_fps[camera_id], 2),
                    'frame_resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                    'gpu_id': gpu_id
                }
                
                # Store result
                self.caption_results[camera_id].append(result_data)
                
                # Output JSON to terminal
                print(f"BLIP Caption: {json.dumps(result_data, indent=2)}")
                
                # Call registered callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(result_data)
                    except Exception as e:
                        logger.error(f"BLIP callback error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame for BLIP captioning {camera_id}: {e}")
    
    def get_latest_captions(self, camera_id: str, count: int = 5) -> List[Dict]:
        """Get latest caption results for a camera"""
        results = list(self.caption_results[camera_id])
        return results[-count:] if results else []
    
    def get_all_latest_captions(self, count: int = 1) -> Dict[str, List[Dict]]:
        """Get latest captions for all cameras"""
        all_results = {}
        for camera_id in self.caption_results:
            all_results[camera_id] = self.get_latest_captions(camera_id, count)
        return all_results
    
    def register_callback(self, callback):
        """Register a callback function to be called when captions are generated"""
        self.result_callbacks.append(callback)
    
    def get_camera_stats(self) -> Dict[str, Dict]:
        """Get statistics for all cameras"""
        stats = {}
        for camera_id in self.camera_fps:
            latest_results = self.get_latest_captions(camera_id, 1)
            stats[camera_id] = {
                'fps': round(self.camera_fps[camera_id], 2),
                'assigned_gpu': self.camera_gpu_mapping.get(camera_id, 'unassigned'),
                'queue_size': self.camera_queues[camera_id].qsize(),
                'total_captions': len(self.caption_results[camera_id]),
                'last_caption_time': latest_results[0]['timestamp'] if latest_results else None,
                'latest_caption': latest_results[0]['caption'] if latest_results else None
            }
        return stats
    
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        for thread in self.camera_threads.values():
            if thread.is_alive():
                thread.join(timeout=2.0)
        logger.info("BLIP captioner stopped")
    

# Global captioner instance
captioner = None

def initialize_captioner(enabled: bool = True):
    """Initialize the global BLIP captioner with fastest model"""
    global captioner
    captioner = BLIPCaptioner(enabled=enabled)
    return captioner

def get_captioner():
    """Get the global BLIP captioner instance"""
    global captioner
    if captioner is None:
        captioner = initialize_captioner()
    return captioner

def process_frame(camera_id: str, frame_data: bytes, timestamp: Optional[float] = None):
    """Process a single frame through BLIP captioning"""
    captioner = get_captioner()
    captioner.add_frame(camera_id, frame_data, timestamp)

def get_captions(camera_id: str, count: int = 5):
    """Get latest caption results for a camera"""
    captioner = get_captioner()
    return captioner.get_latest_captions(camera_id, count)

def get_all_captions(count: int = 1):
    """Get latest captions for all cameras"""
    captioner = get_captioner()
    return captioner.get_all_latest_captions(count)

def get_stats():
    """Get camera captioning statistics"""
    captioner = get_captioner()
    return captioner.get_camera_stats()

def get_current_model_info():
    """Get current model information"""
    captioner = get_captioner()
    return {
        "model_name": captioner.model_name,
        "model_type": captioner.model_type,
        "enabled": captioner.enabled
    }

if __name__ == "__main__":
    # Test the captioner
    captioner = initialize_captioner()
    print("BLIP captioner initialized with multi-GPU support")
    print(f"Available models on GPUs: {list(captioner.models.keys())}")
    
    # Keep the script running for testing
    try:
        while True:
            time.sleep(1)
            stats = captioner.get_camera_stats()
            if stats:
                print(f"\nBLIP Caption Stats: {json.dumps(stats, indent=2)}")
    except KeyboardInterrupt:
        captioner.stop()
        print("\nShutting down BLIP captioner")