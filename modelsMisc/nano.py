import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq  # custom class from remote code

MODEL_ID = "lusxvr/nanoVLM-222M"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Force-load custom model with trust_remote_code
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
model.eval()

cap = cv2.VideoCapture(0)
frame_count = 0
CAP_INTERVAL = 30
caption = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % CAP_INTERVAL == 0:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Caption:", caption)

    cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.imshow("nanoVLM 222M Captioning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
