import cv2
import time
from datetime import datetime
from ultralytics import YOLO

CAMERA_INDEX = 1

class YOLOModel:
    def __init__(self):
        # Load model (nano version is fastest and smallest)
        # self.model = YOLO("./modelsYolo/yolov8x.pt")
        # self.model = YOLO("./modelsYolo/yolov8l.pt")
        # self.model = YOLO("./modelsYolo/yolov8m.pt")
        # self.model = YOLO("./modelsYolo/yolov8s.pt")
        # self.model = YOLO("./modelsYolo/yolov8n.pt")
        self.model = YOLO("./modelsYolo/yolo11s.pt")
        # self.model = YOLO("./modelsYolo/yolo11n-seg.pt")
        # self.model = YOLO("./modelsYolo/yolo11n-pose.pt")
        
        # Person tracking variables
        self.person_id_counter = 0
        self.person_tracks = {}  # Store person positions for ID assignment
        self.next_person_id = 1

    def process_frame(self, frame):
        """Process a frame and return detection results"""
        results = self.model(frame, verbose=False)[0]
        labels = [self.model.names[int(cls)] for cls in results.boxes.cls]
        return results, labels

    def assign_person_ids(self, results):
        """Assign IDs to detected persons based on position"""
        person_detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[int(cls_id)]
                
                if class_name == "person":
                    # Calculate center of bounding box
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Find closest existing person track
                    min_distance = float('inf')
                    assigned_id = None
                    
                    for person_id, (track_x, track_y) in self.person_tracks.items():
                        distance = ((center_x - track_x) ** 2 + (center_y - track_y) ** 2) ** 0.5
                        if distance < min_distance and distance < 100:  # Threshold for same person
                            min_distance = distance
                            assigned_id = person_id
                    
                    # If no close match found, assign new ID
                    if assigned_id is None:
                        assigned_id = self.next_person_id
                        self.next_person_id += 1
                    
                    # Update track
                    self.person_tracks[assigned_id] = (center_x, center_y)
                    
                    person_detections.append({
                        'id': assigned_id,
                        'bbox': box,
                        'confidence': conf,
                        'center': (center_x, center_y)
                    })
        
        return person_detections

    def run_standalone(self):
        """Run YOLO as a standalone application"""
        # Open webcam
        cap = cv2.VideoCapture(CAMERA_INDEX)

        # FPS calculation variables
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 30:  # Update FPS every 30 frames
                current_time = time.time()
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time

            results, labels = self.process_frame(frame)

            # Get person detections with IDs
            person_detections = self.assign_person_ids(results)

            # Log format: H:M:S - object1, object2, ...
            timestamp = datetime.now().strftime("%H:%M:%S")
            person_count = len(person_detections)
            print(f"{timestamp} - {', '.join(labels)} (Persons: {person_count})")

            # Visual feedback
            annotated = results.plot()
            
            # Add FPS overlay
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add person count overlay
            person_count_text = f"Persons: {person_count}"
            cv2.putText(annotated, person_count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw person IDs on bounding boxes
            for person in person_detections:
                x1, y1, x2, y2 = map(int, person['bbox'])
                person_id = person['id']
                
                # Draw ID on bounding box
                id_text = f"ID: {person_id}"
                cv2.putText(annotated, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow("YOLOv8 Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo = YOLOModel()
    yolo.run_standalone()