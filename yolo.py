import cv2
from datetime import datetime
from ultralytics import YOLO

class YOLOModel:
    def __init__(self):
        # Load model (nano version is fastest and smallest)
        # self.model = YOLO("./yoloModels/yolov8x.pt")
        self.model = YOLO("./yoloModels/yolov8l.pt")
        # self.model = YOLO("./yoloModels/yolov8m.pt")
        # self.model = YOLO("./yoloModels/yolov8s.pt")
        # self.model = YOLO("./yoloModels/yolov8n.pt")

    def process_frame(self, frame):
        """Process a frame and return detection results"""
        results = self.model(frame, verbose=False)[0]
        labels = [self.model.names[int(cls)] for cls in results.boxes.cls]
        return results, labels

    def run_standalone(self):
        """Run YOLO as a standalone application"""
        # Open webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results, labels = self.process_frame(frame)

            # Log format: H:M:S - object1, object2, ...
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} - {', '.join(labels)}")

            # Visual feedback
            annotated = results.plot()
            cv2.imshow("YOLOv8 Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo = YOLOModel()
    yolo.run_standalone()
