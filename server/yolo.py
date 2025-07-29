import cv2
import numpy as np
import torch
import threading
import time
import json
from collections import deque, defaultdict
from datetime import datetime
from ultralytics import YOLO
import queue
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path: str = "yolo11m.pt", confidence_threshold: float = 0.5, draw_boxes: bool = True, enabled: bool = True):
        """
        Initialize YOLO detector with multi-GPU support
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            draw_boxes: Whether to draw bounding boxes on frames
            enabled: Whether YOLO processing is enabled
        """
        self.confidence_threshold = confidence_threshold
        self.detection_interval = 1.0  # Process every 1 second for efficiency
        self.draw_boxes = draw_boxes
        self.enabled = enabled
        
        # GPU configuration - distribute across 8 Tesla V100s
        self.available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
        logger.info(f"Available GPUs: {self.available_gpus}")
        
        # Load models on different GPUs for load balancing
        self.models = {}
        for i, gpu_id in enumerate(self.available_gpus[:8]):  # Use up to 8 GPUs
            if gpu_id is not None:
                device = f"cuda:{gpu_id}"
                model = YOLO(model_path)
                model.to(device)
                self.models[gpu_id] = model
                logger.info(f"Loaded YOLO model on GPU {gpu_id}")
            else:
                # CPU fallback
                self.models[0] = YOLO(model_path)
                logger.info("Loaded YOLO model on CPU")
                break
        
        # Camera-specific data structures
        self.camera_queues = defaultdict(lambda: queue.Queue(maxsize=10))
        self.camera_threads = {}
        self.camera_gpu_mapping = {}
        self.detection_results = defaultdict(lambda: deque(maxlen=100))
        self.camera_fps = defaultdict(float)
        self.camera_last_detection = defaultdict(float)
        self.frame_buffers = defaultdict(lambda: deque(maxlen=60))  # Buffer frames for FPS calculation
        self.annotated_frames = defaultdict(bytes)  # Store latest annotated frames
        
        # Thread management
        self.running = True
        self.result_callbacks = []
        
    def assign_gpu_to_camera(self, camera_id: str) -> int:
        """Assign a GPU to a camera for load balancing"""
        if camera_id not in self.camera_gpu_mapping:
            # Round-robin GPU assignment
            gpu_id = len(self.camera_gpu_mapping) % len(self.models)
            available_gpu_ids = list(self.models.keys())
            self.camera_gpu_mapping[camera_id] = available_gpu_ids[gpu_id]
            logger.info(f"Assigned camera {camera_id} to GPU {self.camera_gpu_mapping[camera_id]}")
        return self.camera_gpu_mapping[camera_id]
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: List[Dict], model) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input frame as numpy array
            detections: List of detection dictionaries
            model: YOLO model for accessing class names
            
        Returns:
            Annotated frame
        """
        if not self.draw_boxes or not detections:
            return frame
            
        annotated_frame = frame.copy()
        
        # Define colors for different classes (cycling through a palette)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203), # Pink
            (165, 42, 42),  # Brown
        ]
        
        for detection in detections:
            if detection['bbox'] is None:
                continue
                
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        return annotated_frame
    
    def add_frame(self, camera_id: str, frame_data: bytes, timestamp: Optional[float] = None):
        """
        Add frame to processing queue
        
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
        
        # Skip YOLO processing if disabled
        if not self.enabled:
            return
        
        # Only process if enough time has passed (detection_interval)
        last_detection = self.camera_last_detection[camera_id]
        if timestamp - last_detection >= self.detection_interval:
            try:
                self.camera_queues[camera_id].put_nowait({
                    'frame_data': frame_data,
                    'timestamp': timestamp,
                    'camera_id': camera_id
                })
                self.camera_last_detection[camera_id] = timestamp
                
                # Start processing thread if not already running
                if camera_id not in self.camera_threads or not self.camera_threads[camera_id].is_alive():
                    self.camera_threads[camera_id] = threading.Thread(
                        target=self._process_camera_queue,
                        args=(camera_id,),
                        daemon=True
                    )
                    self.camera_threads[camera_id].start()
                    
            except queue.Full:
                logger.warning(f"Queue full for camera {camera_id}, dropping frame")
    
    def _process_camera_queue(self, camera_id: str):
        """Process frames for a specific camera"""
        gpu_id = self.assign_gpu_to_camera(camera_id)
        model = self.models[gpu_id]
        
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
                
                # Run YOLO detection
                results = model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Extract detection information
                detections = []
                for result in results:
                    for box in result.boxes:
                        if box.conf is not None and box.cls is not None:
                            detection = {
                                'class_id': int(box.cls.cpu().numpy()),
                                'class_name': model.names[int(box.cls.cpu().numpy())],
                                'confidence': float(box.conf.cpu().numpy()),
                                'bbox': box.xyxy.cpu().numpy().tolist()[0] if box.xyxy is not None else None
                            }
                            detections.append(detection)
                
                # Draw bounding boxes on frame
                annotated_frame = self.draw_bounding_boxes(frame, detections, model)
                
                # Encode annotated frame as JPEG
                _, annotated_jpg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_bytes = annotated_jpg.tobytes()
                
                # Store annotated frame for video streaming
                self.annotated_frames[camera_id] = annotated_bytes
                
                # Prepare result
                result_data = {
                    'camera_name': camera_id,
                    'objects': detections,
                    'timestamp': datetime.fromtimestamp(frame_info['timestamp']).isoformat(),
                    'unix_timestamp': frame_info['timestamp'],
                    'fps': round(self.camera_fps[camera_id], 2),
                    'detection_count': len(detections),
                    'frame_resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                    'gpu_id': gpu_id
                }
                
                # Store result
                self.detection_results[camera_id].append(result_data)
                
                # Output JSON to terminal
                print(json.dumps(result_data, indent=2))
                
                # Call registered callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(result_data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame for camera {camera_id}: {e}")
    
    def get_latest_detections(self, camera_id: str, count: int = 10) -> List[Dict]:
        """Get latest detection results for a camera"""
        results = list(self.detection_results[camera_id])
        return results[-count:] if results else []
    
    def get_all_latest_detections(self, count: int = 1) -> Dict[str, List[Dict]]:
        """Get latest detections for all cameras"""
        all_results = {}
        for camera_id in self.detection_results:
            all_results[camera_id] = self.get_latest_detections(camera_id, count)
        return all_results
    
    def register_callback(self, callback):
        """Register a callback function to be called when detections are made"""
        self.result_callbacks.append(callback)
    
    def get_annotated_frame(self, camera_id: str) -> Optional[bytes]:
        """Get the latest annotated frame for a camera"""
        return self.annotated_frames.get(camera_id)
    
    def get_camera_stats(self) -> Dict[str, Dict]:
        """Get statistics for all cameras"""
        stats = {}
        for camera_id in self.camera_fps:
            latest_results = self.get_latest_detections(camera_id, 1)
            stats[camera_id] = {
                'fps': round(self.camera_fps[camera_id], 2),
                'assigned_gpu': self.camera_gpu_mapping.get(camera_id, 'unassigned'),
                'queue_size': self.camera_queues[camera_id].qsize(),
                'total_detections': len(self.detection_results[camera_id]),
                'last_detection_time': latest_results[0]['timestamp'] if latest_results else None,
                'recent_object_count': latest_results[0]['detection_count'] if latest_results else 0
            }
        return stats
    
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        for thread in self.camera_threads.values():
            if thread.is_alive():
                thread.join(timeout=2.0)
        logger.info("YOLO detector stopped")

# Global detector instance
detector = None

def initialize_detector(model_path: str = "yolo11m.pt", confidence_threshold: float = 0.5, draw_boxes: bool = True, enabled: bool = True):
    """Initialize the global YOLO detector"""
    global detector
    detector = YOLODetector(model_path, confidence_threshold, draw_boxes, enabled)
    return detector

def get_detector():
    """Get the global YOLO detector instance"""
    global detector
    if detector is None:
        detector = initialize_detector()
    return detector

def process_frame(camera_id: str, frame_data: bytes, timestamp: Optional[float] = None):
    """Process a single frame through YOLO detection"""
    detector = get_detector()
    detector.add_frame(camera_id, frame_data, timestamp)

def get_detections(camera_id: str, count: int = 10):
    """Get latest detection results for a camera"""
    detector = get_detector()
    return detector.get_latest_detections(camera_id, count)

def get_all_detections(count: int = 1):
    """Get latest detections for all cameras"""
    detector = get_detector()
    return detector.get_all_latest_detections(count)

def get_stats():
    """Get camera processing statistics"""
    detector = get_detector()
    return detector.get_camera_stats()

def get_annotated_frame(camera_id: str):
    """Get the latest annotated frame for a camera"""
    detector = get_detector()
    return detector.get_annotated_frame(camera_id)

if __name__ == "__main__":
    # Test the detector
    detector = initialize_detector()
    print("YOLO detector initialized with multi-GPU support")
    print(f"Available models on GPUs: {list(detector.models.keys())}")
    
    # Keep the script running for testing
    try:
        while True:
            time.sleep(1)
            stats = detector.get_camera_stats()
            if stats:
                print(f"\nCamera Stats: {json.dumps(stats, indent=2)}")
    except KeyboardInterrupt:
        detector.stop()
        print("\nShutting down YOLO detector")