import cv2
import threading
import time
from models.blip import BLIPModel
from models.yolo import YOLOModel

class CameraStreamWeb:
    def __init__(self, camera_config):
        self.id = camera_config["id"]
        self.index = camera_config["index"]
        self.blip_model = BLIPModel()
        self.yolo_model = YOLOModel()
        if isinstance(self.index, str) and self.index.startswith('rtsp://'):
            self.blip_model.caption_interval = 5
        else:
            self.blip_model.caption_interval = 15
        self.current_caption = ""
        self.current_objects = []
        self.frame_count = 0
        self.cap = None
        self.running = False
        self.latest_annotated = None
        self.lock = threading.Lock()

    def initialize_camera(self):
        if isinstance(self.index, str) and self.index.startswith('rtsp://'):
            self.cap = cv2.VideoCapture(self.index)
        else:
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.index} ({self.id})")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def add_caption_overlay(self, frame, caption):
        if not caption:
            return frame
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
        lines = lines[:3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25
        padding = 10
        total_height = len(lines) * line_height + 2 * padding
        bg_x1 = 10
        bg_y1 = 10
        bg_x2 = 630
        bg_y2 = bg_y1 + total_height
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        y_position = bg_y1 + padding + 20
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (bg_x1 + 5, y_position), font,
                        font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            y_position += line_height
        return frame

    def process_frame(self, frame):
        self.frame_count += 1
        results, labels = self.yolo_model.process_frame(frame)
        self.current_objects = labels
        caption = self.blip_model.process_frame(frame)
        if caption:
            self.current_caption = caption
        annotated = results.plot()
        annotated = cv2.resize(annotated, (640, 480), interpolation=cv2.INTER_AREA)
        annotated = self.add_caption_overlay(annotated, self.current_caption)
        return annotated

    def run_stream(self):
        if not self.initialize_camera():
            return
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            annotated = self.process_frame(frame)
            with self.lock:
                self.latest_annotated = annotated
        self.cap.release()

    def get_jpeg(self):
        with self.lock:
            frame = self.latest_annotated.copy() if self.latest_annotated is not None else None
        if frame is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return jpeg.tobytes() 