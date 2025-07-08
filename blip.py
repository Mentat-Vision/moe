import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import warnings
import os
import gc

# Suppress warnings and verbose output
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_grad_enabled(False)  # Disable gradients globally

class BLIPModel:
    def __init__(self):
        # Optimized BLIP configuration
        self.model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Use half precision for speed
            low_cpu_mem_usage=True
        )

        # Optimized device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()

        # Memory optimization
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        self.frame_count = 0
        self.caption_interval = 45  # Optimized interval for better performance
        self.current_caption = ""

    def process_frame(self, frame):
        """Process a frame and return caption if ready"""
        self.frame_count += 1
        
        if self.frame_count % self.caption_interval == 0:
            # Optimized frame processing
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Process with optimized settings
            inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=25,  # Balanced length for detail
                    num_beams=3,    # Small beam search for better quality
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.2,  # Prevent repetitive text
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            self.current_caption = caption
            
            # Memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return caption
        return None

    def run_standalone(self):
        """Run BLIP as a standalone application"""
        # Optimized webcam settings
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            caption = self.process_frame(frame)
            
            if caption:
                # Clean timestamp output
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"{timestamp} - {caption}")

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
                
                # Display up to 3 lines
                y_position = 30
                for i, line in enumerate(lines[:3]):
                    cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    y_position += 25

            cv2.imshow("Optimized BLIP Captioning", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Final cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    blip = BLIPModel()
    blip.run_standalone()
