import cv2
import torch
from datetime import datetime
import warnings
import os
import gc
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import threading
import time

# Suppress warnings and verbose output
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_grad_enabled(False)

# Configuration - Uncomment the logging mode you prefer
LOG_MODE = "per_microsecond"  # Options: "per_microsecond" or "per_frame"
# LOG_MODE = "per_frame"

class UnifiedModelSystem:
    def __init__(self):
        # Initialize BLIP
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.model_name, use_fast=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Initialize YOLO
        self.yolo_model = YOLO("yolov8n.pt")
        
        # Device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.blip_model.to(self.device)
        self.blip_model.eval()
        
        # State variables
        self.current_caption = ""
        self.current_objects = []
        self.frame_count = 0
        self.caption_interval = 45
        self.last_log_time = 0
        self.log_interval = 0.1  # 100ms for microsecond logging
        
        # Memory optimization
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    def process_frame(self, frame):
        """Process a single frame with both models"""
        self.frame_count += 1
        
        # Process with YOLO (every frame)
        results = self.yolo_model(frame, verbose=False)[0]
        labels = [self.yolo_model.names[int(cls)] for cls in results.boxes.cls]
        self.current_objects = labels
        
        # Process with BLIP (every caption_interval frames)
        if self.frame_count % self.caption_interval == 0:
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(
                    **inputs,
                    max_length=25,
                    num_beams=3,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            self.current_caption = caption
            
            # Memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return results
    
    def should_log(self):
        """Determine if we should log based on the selected mode"""
        current_time = time.time()
        
        if LOG_MODE == "per_microsecond":
            # Log every 100ms (microsecond precision)
            if current_time - self.last_log_time >= self.log_interval:
                self.last_log_time = current_time
                return True
            return False
        elif LOG_MODE == "per_frame":
            # Log when we have new caption or objects (original behavior)
            return (self.frame_count % self.caption_interval == 0 or self.current_objects)
        else:
            return False
    
    def get_unified_log(self):
        """Get the unified log entry"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        caption = self.current_caption if self.current_caption else "No caption yet"
        objects = ', '.join(self.current_objects) if self.current_objects else "No objects detected"
        
        return f"{timestamp}\n - caption: {caption}\n - object list: {objects}"
    
    def run(self):
        """Main run loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"Starting Unified Model System with {LOG_MODE} logging...")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with both models
                results = self.process_frame(frame)
                
                # Print unified log based on selected mode
                if self.should_log():
                    print(self.get_unified_log())
                    print()  # Empty line for readability
                
                # Create annotated frame
                annotated = results.plot()
                
                # Add caption overlay if available
                if self.current_caption:
                    words = self.current_caption.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) < 40:
                            current_line += (" " + word) if current_line else word
                        else:
                            lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    
                    y_position = 30
                    for i, line in enumerate(lines[:3]):
                        cv2.putText(annotated, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        y_position += 25
                
                cv2.imshow("Unified Model System", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    system = UnifiedModelSystem()
    system.run() 