import cv2
from datetime import datetime
from ultralytics import YOLO

# Load model (nano version is fastest and smallest)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    labels = [model.names[int(cls)] for cls in results.boxes.cls]

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
