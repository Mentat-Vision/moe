import cv2
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Use MPS device if available (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load Kosmos-2 model and processor
model_id = "microsoft/kosmos-2-patch14-224"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id).to(device)

# Capture image from webcam immediately
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image.")
    exit()

# Convert to RGB and create PIL Image
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = Image.fromarray(frame_rgb)

# Prompt for Kosmos
prompt = "Describe this image in detail."

# Run inference
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, max_new_tokens=256)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Output
print("\nCaption:")
print(caption)

# Optional: show image in a window
cv2.imshow("Captured Image", frame)
cv2.waitKey(3000)  # Show for 3 seconds
cv2.destroyAllWindows()
