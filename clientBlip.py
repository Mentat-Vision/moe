import cv2
import requests
import base64
import numpy as np
from datetime import datetime
import threading
import time
import queue

CAMERA_INDEX = 2
SERVER_URL = "http://10.8.162.58:5001/caption"  # Update with your Sampo IP

class BLIPClient:
    def __init__(self):
        self.frame_count = 0
        self.caption_interval = 45  # Optimized interval for better performance
        self.current_caption = ""
        self.caption_queue = queue.Queue()
        self.processing = False
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._process_frames_background, daemon=True)
        self.processing_thread.start()

    def _encode_frame(self, frame):
        """Encode frame to base64"""
        # Resize frame to reduce data size
        frame_resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _send_frame_to_server(self, frame):
        """Send frame to Sampo server and get caption"""
        try:
            image_data = self._encode_frame(frame)
            payload = {"image": image_data}
            
            response = requests.post(SERVER_URL, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get("caption", "")
            else:
                print(f"Server error: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Error sending frame to server: {e}")
            return ""

    def _process_frames_background(self):
        """Background thread to process frames"""
        while True:
            if not self.processing:
                time.sleep(0.1)
                continue
                
            try:
                # Get frame from queue
                frame = self.caption_queue.get(timeout=1)
                caption = self._send_frame_to_server(frame)
                if caption:
                    self.current_caption = caption
                    # Clean timestamp output
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"{timestamp} - {caption}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Background processing error: {e}")

    def process_frame(self, frame):
        """Process a frame and return caption if ready"""
        self.frame_count += 1
        
        if self.frame_count % self.caption_interval == 0:
            # Add frame to processing queue
            self.caption_queue.put(frame.copy())
            self.processing = True
            
        return self.current_caption

    def run_standalone(self):
        """Run BLIP client as a standalone application"""
        # Optimized webcam settings
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time

        print("🔁 BLIP Client running. Press 'q' to quit.")
        print(f"📡 Sending frames to server: {SERVER_URL}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            caption = self.process_frame(frame)

            # Optimized caption overlay
            if self.current_caption:
                # Word wrapping for better display
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
                
                # Display up to 3 lines with black background
                y_position = 30
                for i, line in enumerate(lines[:3]):
                    # Get text size to create background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw black background rectangle
                    cv2.rectangle(frame, (10, y_position - text_height - 5), 
                                (10 + text_width + 10, y_position + 5), (0, 0, 0), -1)
                    
                    # Draw text on top of background
                    cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    y_position += 25

            cv2.imshow("BLIP Client - Remote Processing", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.processing = False

if __name__ == "__main__":
    blip_client = BLIPClient()
    blip_client.run_standalone() 