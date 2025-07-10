import cv2
import threading
import time
from datetime import datetime
import warnings
import os
import queue
import json

# Import our modular components
from blip import BLIPModel
from yolo import YOLOModel

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
LOG_MODE = "per_microsecond"  # Options: "per_microsecond" or "per_frame"
# LOG_MODE = "per_frame"
SAVE_LOGS_TO_FILE = True  # Set to True to save logs to logs.txt

# Camera Configuration - Just comment/uncomment to enable/disable
# Use integer index for local webcams, RTSP URL for IP cameras
CAMERAS = [
    {"id": "CAM1 - Mac Webcam", "index": 2}, 
    {"id": "CAM2 - USB Webcam", "index": 0}, 
    # {"id": "IP101 - Front Door", "index": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/101/"}, 
    # {"id": "IP201 - Back Door", "index": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/201/"},  
    # {"id": "IP301 - Side Door", "index": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/301/"},  
    # {"id": "IP401 - Stairs", "index": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/401/"}, 
    # {"id": "IP501 - Parking Outwards", "index": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/501/"},  
    # {"id": "IP601 - Parking Towards", "index": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/601/"}, 
]

class CameraStream:
    """Individual camera stream with its own BLIP and YOLO models"""
    
    def __init__(self, camera_config):
        self.id = camera_config["id"]
        self.index = camera_config["index"]
        
        # Initialize models for this camera
        self.blip_model = BLIPModel()
        self.yolo_model = YOLOModel()
        
        # Different caption intervals for different camera types
        if isinstance(self.index, str) and self.index.startswith('rtsp://'):
            # IP cameras - more frequent captioning
            self.blip_model.caption_interval = 5  # Every 5 frames for IP cameras
        else:
            # Local webcams - standard interval
            self.blip_model.caption_interval = 15  # Every 15 frames for local cameras
        
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
        # Check if this is an IP camera (RTSP URL) or local webcam
        if isinstance(self.index, str) and self.index.startswith('rtsp://'):
            # IP camera - use RTSP URL
            self.cap = cv2.VideoCapture(self.index)
            print(f"Connecting to IP camera: {self.index}")
        else:
            # Local webcam - use integer index
            self.cap = cv2.VideoCapture(self.index)
            print(f"Connecting to local camera: {self.index}")
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.index} ({self.id})")
            return False
            
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Additional settings for IP cameras (without the problematic property)
        if isinstance(self.index, str) and self.index.startswith('rtsp://'):
            # RTSP specific settings for better performance
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            # Try to set TCP preference if available
            try:
                self.cap.set(cv2.CAP_PROP_PROTOCOL_PREFERENCE, 0)  # TCP for better reliability
            except AttributeError:
                # Property not available, skip it
                pass
        
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
        """Get the log entry for this camera in JSON format"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        caption = self.current_caption if self.current_caption else "No caption yet"
        objects = self.current_objects if self.current_objects else []
        
        log_data = {
            "timestamp": timestamp,
            "camera": self.id,
            "caption": caption,
            "objects": objects
        }
        
        log_entry = json.dumps(log_data, indent=2)
        
        # Save to file if enabled
        if SAVE_LOGS_TO_FILE:
            try:
                with open('logs.txt', 'a') as f:
                    f.write(log_entry + '\n\n')
            except Exception as e:
                pass  # Silently fail if file writing fails
        
        return log_entry
    
    def add_caption_overlay(self, frame, caption):
        """Add caption overlay with black background for better readability"""
        if not caption:
            return frame
            
        # Word wrapping for better display
        words = caption.split()
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
        
        # Limit to 3 lines
        lines = lines[:3]
        
        # Calculate text dimensions and background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25
        padding = 10
        
        # Calculate total height needed
        total_height = len(lines) * line_height + 2 * padding
        
        # Create black background rectangle
        bg_x1 = 10
        bg_y1 = 10
        bg_x2 = 630  # Leave some margin from right edge
        bg_y2 = bg_y1 + total_height
        
        # Draw black background with some transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Blend the overlay with the original frame (70% original, 30% black)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add text
        y_position = bg_y1 + padding + 20  # Start text below padding
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (bg_x1 + 5, y_position), font,
                        font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            y_position += line_height
        
        return frame
    
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
                    # For IP cameras, try to reconnect
                    if isinstance(self.index, str) and self.index.startswith('rtsp://'):
                        print(f"Attempting to reconnect to {self.id}...")
                        time.sleep(2)
                        self.cap.release()
                        if self.initialize_camera():
                            continue
                    break
                
                # Resize frame to standard size (640x480) for consistent display
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                
                # Process frame with both models
                results = self.process_frame(frame)
                
                # Print log if needed
                if self.should_log():
                    print(self.get_log_entry())
                    print()  # Empty line for readability
                
                # Create annotated frame
                annotated = results.plot()
                
                # Resize annotated frame to ensure consistent display size
                annotated = cv2.resize(annotated, (640, 480), interpolation=cv2.INTER_AREA)
                
                # Add caption overlay with black background
                annotated = self.add_caption_overlay(annotated, self.current_caption)
                
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
        
        # Clear logs file at startup
        if SAVE_LOGS_TO_FILE:
            try:
                with open('logs.txt', 'w') as f:
                    f.write('')  # Clear the file
                print("Logs file cleared for new session.")
            except Exception as e:
                print(f"Warning: Could not clear logs file: {e}")
        
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