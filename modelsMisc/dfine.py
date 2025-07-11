import cv2
import torch
from transformers import DFineForObjectDetection, AutoImageProcessor

# Config
CAM_INDEX = 1
CONF_THRESH = 0.5   # confidence threshold

# Load model + processor
processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-large-coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-large-coco")
model.eval()

# Start video capture
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera {CAM_INDEX}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare image for model (RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=[img.shape[:2]], threshold=CONF_THRESH
    )[0]

    # Draw boxes and labels
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = model.config.id2label[label_id.item()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    # Show result
    cv2.imshow("D-FINE Webcam", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
