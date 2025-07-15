import cv2
import numpy as np
import os
from ultralytics import YOLO
from .baseWorker import BaseWorker

class YOLOWorker(BaseWorker):
    """YOLO expert worker that processes object detection jobs"""
    
    def __init__(self, config):
        super().__init__("YOLO", config)
        self.model = None
    
    async def initialize_model(self):
        """Initialize the YOLO model"""
        model_path = self.config.get("YOLO_MODEL_PATH")
        
        if not model_path:
            raise Exception("YOLO_MODEL_PATH not configured in config.env")
        
        if not os.path.exists(model_path):
            print(f"❌ YOLO model not found at {model_path}")
            raise Exception(f"YOLO model not found: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
                
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise e
    
    async def process_frame(self, job):
        """Process a frame with YOLO object detection"""
        try:
            frame = job["frame"]
            camera_id = job["camera_id"]
            
            if self.model is None:
                return {"error": "YOLO model not loaded"}
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Extract detections
            detections = []
            person_detections = []
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "class": class_name,
                            "confidence": confidence,
                            "class_id": class_id
                        }
                        detections.append(detection)
                        
                        # Count persons
                        if class_name.lower() == "person":
                            person_count += 1
                            person_detections.append(detection)
            
            # Get current stats
            stats = self.get_stats()
            
            return {
                "detections": detections,
                "person_detections": person_detections,
                "person_count": person_count,
                "fps": stats["fps"],
                "camera_id": camera_id
            }
            
        except Exception as e:
            print(f"❌ YOLO Worker error processing frame: {e}")
            return {
                "error": str(e),
                "detections": [],
                "person_detections": [],
                "person_count": 0,
                "fps": 0,
                "camera_id": job.get("camera_id", 0)
            } 