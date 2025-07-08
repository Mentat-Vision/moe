import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Load model and processor
model_id = "microsoft/kosmos-2-patch14-224"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id).to("mps")  # or "cuda" if you're on Windows/Linux with GPU

# Load your image (replace with webcam/frame later)
# image = Image.open("test.png")
image = Image.open("1.JPG")

# Prompt
prompt = "Describe this image in detail."

# Process input
inputs = processor(text=prompt, images=image, return_tensors="pt").to("mps")

# Generate caption
generated_ids = model.generate(**inputs, max_new_tokens=256)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Caption:", caption)
