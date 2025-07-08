import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image

# Change this path to your test image
# image_path = "test.png"
image_path = "1.JPG"

# Load image
image = Image.open(image_path).convert('RGB')

# Load the model and processor
model_id = "Salesforce/instructblip-flan-t5-xl"
device = "mps" if torch.backends.mps.is_available() else "cpu"  # macOS M1/M2 support

print("Loading model...")
processor = InstructBlipProcessor.from_pretrained(model_id)
model = InstructBlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16 if device == "mps" else torch.float32)
model.to(device)

# Prompt (can be modified)
prompt = "Describe this image in detail."

# Preprocess input
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16 if device == "mps" else torch.float32)

# Generate caption
print("Generating...")
output = model.generate(**inputs, max_new_tokens=100)
caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)

print("\n📸 Caption:")
print(caption)
