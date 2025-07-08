import cv2
import threading
import time
from datetime import datetime
import warnings
import os
import queue

# Import our modular components
from blip import BLIPModel
from yolo import YOLOModel

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
LOG_MODE = "per_microsecond"  # Options: "per_microsecond" or "per_frame"
# LOG_MODE = "per_frame"

# Camera Configuration - Just comment/uncomment to enable/disable
CAMERAS = [
    {"id": "CAM1", "index": 0},  
    {"id": "CAM2", "index": 2},  
]

class CameraStream:
    """Individual camera stream with its own BLIP and YOLO models"""
    
    def __init__(self, camera_config):
        self.id = camera_config["id"]
        self.index = camera_config["index"]
        
        # Initialize models for this camera
        self.blip_model = BLIPModel()
        self.yolo_model = YOLOModel()
        
        # State variables
        self.current_caption = ""
        self.current_objects = []
        self.frame_count = 0
        self.last_log_time = 0
        self.log_interval = 0.1  # 100ms for microsecond logging
        
        # Camera capture
        self.cap = None
        self.running = False
        
        # Frame queue for display
        self.frame_queue = queue.Queue(maxsize=1)
        
    def initialize_camera(self):
        """Initialize the camera capture"""
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.index} ({self.id})")
            return False
            
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    def process_frame(self, frame):
        """Process a single frame with both models"""
        self.frame_count += 1
        
        # Process with YOLO (every frame)
        results, labels = self.yolo_model.process_frame(frame)
        self.current_objects = labels
        
        # Process with BLIP (every caption_interval frames)
        caption = self.blip_model.process_frame(frame)
        if caption:
            self.current_caption = caption
        
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
            # Log when we have new caption or objects
            return (self.frame_count % self.blip_model.caption_interval == 0 or self.current_objects)
        else:
            return False
    
    def get_log_entry(self):
        """Get the log entry for this camera"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        caption = self.current_caption if self.current_caption else "No caption yet"
        objects = ', '.join(self.current_objects) if self.current_objects else "No objects detected"
        
        return f"{self.id}\n - {timestamp}\n - Caption: {caption}\n - Objects: {objects}"
    
    def run_stream(self):
        """Run the camera stream in its own thread"""
        if not self.initialize_camera():
            return
            
        print(f"Starting {self.id}...")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Error reading from {self.id}")
                    break
                
                # Process frame with both models
                results = self.process_frame(frame)
                
                # Print log if needed
                if self.should_log():
                    print(self.get_log_entry())
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
                
                # Put frame in queue for display (non-blocking)
                try:
                    self.frame_queue.put_nowait(annotated)
                except queue.Full:
                    # Skip frame if queue is full
                    pass
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.cap:
            self.cap.release()

class MultiCameraSystem:
    """Main system that manages multiple camera streams"""
    
    def __init__(self):
        self.cameras = []
        self.threads = []
        self.running = False
        
        # Initialize all cameras in the list
        for camera_config in CAMERAS:
            camera = CameraStream(camera_config)
            self.cameras.append(camera)
    
    def start(self):
        """Start all camera streams"""
        if not self.cameras:
            print("No cameras configured. Check CAMERAS list.")
            return
            
        self.running = True
        print(f"Starting Multi-Camera System with {LOG_MODE} logging...")
        print(f"Active cameras: {[cam.id for cam in self.cameras]}")
        print("Press 'q' to quit\n")
        
        # Start each camera in its own thread
        for camera in self.cameras:
            camera.running = True
            thread = threading.Thread(target=camera.run_stream, daemon=True)
            thread.start()
            self.threads.append(thread)
        
        # Main display loop (runs in main thread)
        self.display_loop()
    
    def display_loop(self):
        """Main display loop that handles all camera windows"""
        try:
            while self.running:
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Display frames from all cameras
                for camera in self.cameras:
                    try:
                        frame = camera.frame_queue.get_nowait()
                        cv2.imshow(f"{camera.id}", frame)
                    except queue.Empty:
                        # No frame available, skip
                        pass
                
                # Small delay to prevent high CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all camera streams"""
        self.running = False
        for camera in self.cameras:
            camera.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Cleanup
        for camera in self.cameras:
            camera.cleanup()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = MultiCameraSystem()
    system.start() 