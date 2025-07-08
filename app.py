import cv2
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor
import requests
import argparse
import time
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import warnings
from collections import deque
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
import numpy as np
from SERVER_URL import SERVER_URL
import subprocess
import sys
import signal
import os 

# Suppress warnings
urllib3.disable_warnings(InsecureRequestWarning)
warnings.filterwarnings("ignore")

# Configuration
FRAME_INTERVAL = 10
RESOLUTION = (640, 480)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

GENERATION_CONFIG = {
    "max_length": 30,
    "num_beams": 4,
    "num_return_sequences": 1,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0
}
# Camera Configuration - Comment/uncomment cameras you want to run
CAMERAS = [
    # Local webcams
    # {"device": "MacBook Camera", "webcam": "1"},
    # {"device": "iPhone Camera", "webcam": "0"},
    
    # RTSP streams
    {"device": "IP Camera 101", "webcam": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/101/"}, # Front Door
    # {"device": "IP Camera 201", "webcam": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/201/"}, # Back Door
    # {"device": "IP Camera 301", "webcam": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/301/"}, # Side Door? 
    # {"device": "IP Camera 401", "webcam": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/401/"}, # Stairs
    {"device": "IP Camera 501", "webcam": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/501/"}, # Parking Outwards
    {"device": "IP Camera 601", "webcam": "rtsp://Koy%20Otaniemen%20T:Otaranta123@10.19.55.20:554/Streaming/Channels/601/"}, # Parking Towards
]

class ImageCaptioner:
    def __init__(self):
        print(" Initializing image captioning model...")
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(DEVICE)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        print("✅ Model initialization complete")

    def get_caption(self, image):
        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(DEVICE)
            
            with torch.no_grad():
                output_ids = self.model.generate(pixel_values, **GENERATION_CONFIG)
            
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"❌ Caption error: {e}")
            return "Processing frame..."

class WebcamProcessor:
    def __init__(self, device_name, webcam_id):
        self.device_name = device_name
        self.webcam_id = webcam_id
        self.captioner = ImageCaptioner()
        self.setup_camera()
        self.server_available = self.check_server()
        if self.server_available:
            self.register_device()
        else:
            print("⚠️ Running in local-only mode (no server connection)")
        
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 1000000
        self.caption_history = deque()
        self.buffer_duration_seconds = 120
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.active_triggers = set()

    def precise_trigger_match(self, caption, trigger, threshold=0.7):
        conditions = self._parse_trigger_conditions(trigger)
        if not conditions:
            return False
        
        conditions_met = 0
        for condition in conditions:
            if self._condition_in_caption(condition, caption, threshold):
                conditions_met += 1
        
        return conditions_met == len(conditions)
    
    def _parse_trigger_conditions(self, trigger):
        connectors = ['with', 'wearing', 'and', 'or', 'in', 'on', 'has', 'have', 'holding', 'carrying']
        conditions = []
        words = trigger.lower().split()
        
        current_condition = []
        for word in words:
            if word in connectors and current_condition:
                if current_condition:
                    conditions.append(' '.join(current_condition))
                current_condition = []
            else:
                current_condition.append(word)
        
        if current_condition:
            conditions.append(' '.join(current_condition))
        
        if not conditions:
            conditions = [trigger.lower()]
        
        cleaned_conditions = []
        for condition in conditions:
            condition_words = condition.split()
            filtered_words = [word for word in condition_words 
                            if word not in ['a', 'an', 'the', 'is', 'are', 'was', 'were']]
            if filtered_words:
                cleaned_conditions.append(' '.join(filtered_words))
        
        return cleaned_conditions
    
    def _condition_in_caption(self, condition, caption, threshold=0.7):
        condition_words = condition.lower().split()
        caption_words = caption.lower().split()
        
        words_found = sum(1 for word in condition_words if word in caption_words)
        if words_found == len(condition_words):
            return True
        
        try:
            condition_embedding = self.semantic_similarity_model.encode(condition, convert_to_tensor=True)
            caption_embedding = self.semantic_similarity_model.encode(caption, convert_to_tensor=True)
            similarity = util.cos_sim(condition_embedding, caption_embedding)[0]
            return similarity.item() > threshold
        except Exception as e:
            print(f"❌ Semantic similarity error: {e}")
            return False

    def process_frame(self, image):
        return self.captioner.get_caption(image)

    def check_server(self):
        try:
            response = requests.get(SERVER_URL, timeout=30, verify=False)
            return response.status_code == 200
        except:
            return False

    def try_reconnect(self):
        if not self.server_available and self.reconnect_attempts < self.max_reconnect_attempts:
            print(f"🔄 Attempting to reconnect to server (attempt {self.reconnect_attempts + 1})")
            self.server_available = self.check_server()
            if self.server_available:
                self.register_device()
                print("✅ Successfully reconnected to server")
                self.reconnect_attempts = 0
            else:
                self.reconnect_attempts += 1

    def setup_camera(self):
        print(f"📹 Attempting to access camera: {self.webcam_id}")
        
        if isinstance(self.webcam_id, str) and self.webcam_id.startswith('rtsp://'):
            print("📡 Detected RTSP stream")
            self.cap = cv2.VideoCapture(self.webcam_id)
        else:
            try:
                webcam_index = int(self.webcam_id)
                print(f"📹 Detected webcam index: {webcam_index}")
                self.cap = cv2.VideoCapture(webcam_index)
            except ValueError:
                raise RuntimeError(f"Invalid webcam ID: {self.webcam_id}")
        
        if not self.cap.isOpened():
            if isinstance(self.webcam_id, str) and self.webcam_id.startswith('rtsp://'):
                raise RuntimeError(f"Cannot access RTSP stream: {self.webcam_id}")
            else:
                print(f"❌ Failed to open webcam index {self.webcam_id}")
                
                if self.webcam_id != 0:
                    print("🔄 Trying webcam index 0...")
                    self.cap = cv2.VideoCapture(0)
                    if self.cap.isOpened():
                        print("⚠️ Successfully opened iPhone Continuity Camera (index 0)")
                        self.webcam_id = 0
                    else:
                        raise RuntimeError("Cannot access any webcam")
                else:
                    raise RuntimeError(f"Cannot access webcam index {self.webcam_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            if isinstance(self.webcam_id, str) and self.webcam_id.startswith('rtsp://'):
                raise RuntimeError("RTSP stream opened but cannot read frames")
            else:
                raise RuntimeError("Camera opened but cannot read frames")
        
        print(f"✅ Successfully initialized camera: {self.webcam_id}")
        print(f" Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")
        
        if isinstance(self.webcam_id, str) and self.webcam_id.startswith('rtsp://'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            print("📡 RTSP stream configured")
        else:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            
            if self.webcam_id == 0:
                print("⚠️ Using iPhone Continuity Camera")

    def register_device(self):
        try:
            response = requests.post(
                f'{SERVER_URL}/add_device', 
                data={'device': self.device_name}, 
                timeout=30,
                verify=False
            )
            if response.status_code == 200:
                print("✅ Device registered with server")
                return True
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to register device: {e}")
            self.server_available = False
            return False

    def contact_server(self, caption):
        if self.server_available:
            try:
                response = requests.get(f'{SERVER_URL}/triggers', timeout=30, verify=False)
                if response.status_code == 200:
                    triggers = response.json()
                    
                    currently_matching_triggers = set()
                    for trigger in triggers:
                        if self.precise_trigger_match(caption, trigger):
                            currently_matching_triggers.add(trigger)
                    
                    triggers_to_deactivate = self.active_triggers - currently_matching_triggers
                    triggers_to_activate = currently_matching_triggers - self.active_triggers
                    
                    for trigger in triggers_to_deactivate:
                        requests.post(
                            f'{SERVER_URL}/deactivate_trigger', 
                            data={'device': self.device_name, 'trigger': trigger}, 
                            timeout=30, verify=False
                        )
                    
                    for trigger in triggers_to_activate:
                        requests.post(
                            f'{SERVER_URL}/trigger', 
                            data={'device': self.device_name, 'trigger': trigger}, 
                            timeout=30, verify=False
                        )
                    
                    self.active_triggers = currently_matching_triggers
                        
            except Exception as e:
                if "Connection" in str(e):
                    self.server_available = False
                    print("❌ Lost connection to server, switching to local-only mode")
                    self.try_reconnect()

    def trim_caption_buffer(self):
        cutoff = time.time() - self.buffer_duration_seconds
        while self.caption_history and self.caption_history[0][0] < cutoff:
            self.caption_history.popleft()

    def is_similar(self, a, b, threshold=0.5):
        return SequenceMatcher(None, a, b).ratio() > threshold

    def is_noisy(self, caption):
        return (
            len(caption.split()) < 3 or
            caption in ["processing frame...", "error"] or
            any(char.isdigit() for char in caption)
        )

    def get_clean_captions(self):
        captions = [cap for _, cap in self.caption_history]
        cleaned = []
        for cap in captions:
            if self.is_noisy(cap):
                continue
            if not cleaned or (
                len(cleaned) > 0 and not self.is_similar(cleaned[-1], cap) and cleaned[-1] != cap
            ):
                cleaned.append(cap)
        return cleaned
    
    def get_dominant_captions(self, min_cluster_size=3, eps=0.4):
        cleaned_captions = self.get_clean_captions()
        if len(cleaned_captions) < min_cluster_size:
            return cleaned_captions

        embeddings = self.embedding_model.encode(cleaned_captions)
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='cosine').fit(embeddings)
        labels = clustering.labels_

        clustered = {}
        for label, caption in zip(labels, cleaned_captions):
            if label == -1:
                continue
            clustered.setdefault(label, []).append(caption)

        dominant_captions = []
        for cluster in sorted(clustered.values(), key=lambda c: -len(c)):
            dominant_captions.append(cluster[0])
        
        return dominant_captions

    def run(self):
        frame_count = 0
        latest_caption = "Waiting for caption..."
        print("🎬 Press 'q' to quit")

        window_name = f'Live Feed - {self.device_name}'
        cv2.destroyWindow(window_name)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        display_width, display_height = 720, 540

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame read error - retrying...")
                    time.sleep(0.1)
                    continue

                if not self.server_available and frame_count % 100 == 0:
                    self.try_reconnect()

                if frame_count % FRAME_INTERVAL == 0:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(rgb_frame)
                        caption = self.process_frame(pil_image)
                    except Exception as e:
                        print(f"❌ Caption error: {e}")
                        caption = "Processing frame..."

                    timestamp = time.time()
                    latest_caption = caption.lower()
                    self.caption_history.append((timestamp, latest_caption))
                    self.trim_caption_buffer()

                    if self.server_available and (frame_count % 50 == 0):
                        self.contact_server(latest_caption)

                cv2.putText(frame, latest_caption, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                status = "Server: Connected" if self.server_available else "Server: Local Mode"
                cv2.putText(frame, status, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if self.server_available else (0, 165, 255), 2)
                
                display_frame = cv2.resize(frame, (display_width, display_height))
                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if frame_count % 100 == 0:
                    dominant_captions = self.get_dominant_captions()
                    print("📊 Summary-worthy captions:")
                    for cap in dominant_captions:
                        print(f"→ {cap}")

                    if self.server_available and dominant_captions:
                        try:
                            requests.post(
                                f"{SERVER_URL}/device_summary",
                                json={'device': self.device_name, 'captions': dominant_captions},
                                timeout=30, verify=False
                            )
                        except Exception as e:
                            print(f"❌ Error sending summary captions: {e}")
                            self.server_available = False
                            self.try_reconnect()

                frame_count += 1

        except KeyboardInterrupt:
            print("\n🛑 Stopping gracefully...")
        finally:
            self.cleanup()

    def cleanup(self):
        print(f"🧹 Cleaning up {self.device_name}...")
        
        if self.server_available and self.active_triggers:
            print(f"🔄 Deactivating {len(self.active_triggers)} active triggers...")
            for trigger in list(self.active_triggers):
                try:
                    requests.post(
                        f'{SERVER_URL}/deactivate_trigger', 
                        data={'device': self.device_name, 'trigger': trigger}, 
                        timeout=10, verify=False
                    )
                    print(f"   ✅ Deactivated trigger: {trigger}")
                except Exception as e:
                    print(f"   ❌ Failed to deactivate trigger '{trigger}': {e}")
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.server_available:
            try:
                print(f"📡 Notifying server that '{self.device_name}' is disconnecting...")
                response = requests.post(
                    f'{SERVER_URL}/delete_device', 
                    data={'device': self.device_name}, 
                    timeout=10, verify=False
                )
                if response.status_code == 200:
                    print(f"✅ Device '{self.device_name}' successfully disconnected from server")
                else:
                    print(f"⚠️ Server returned status {response.status_code} for device deletion")
            except Exception as e:
                print(f"⚠️ Could not notify server of disconnection: {e}")
        else:
            print(f"ℹ️ Device '{self.device_name}' was not connected to server")
        
        print(f"✅ Cleanup complete for {self.device_name}")

def main():
    parser = argparse.ArgumentParser(description="Live webcam captioning with precise trigger matching")
    parser.add_argument('--single', action='store_true', help='Run in single camera mode with command line arguments')
    parser.add_argument('--device', help='Device name (e.g., "MacBook Camera") - only used with --single')
    parser.add_argument('--webcam', help='Webcam ID or RTSP URL - only used with --single')
    args = parser.parse_args()

    active_cameras = [cam for cam in CAMERAS if not cam.get('device', '').startswith('#')]
    
    if args.single:
        if not args.device or not args.webcam:
            print("❌ Error: --single mode requires both --device and --webcam arguments")
            print("Example: python app.py --single --device 'MacBook Camera' --webcam 1")
            return
        
        processor = None
        
        def signal_handler(signum, frame):
            print(f"\n Received interrupt signal, stopping {args.device}...")
            if processor:
                try:
                    processor.cleanup()
                except Exception as e:
                    print(f"❌ Error during cleanup: {e}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            print(f"🚀 Starting single camera: {args.device}")
            processor = WebcamProcessor(args.device, args.webcam)
            processor.run()
        except KeyboardInterrupt:
            print(f"\n Stopping {args.device}...")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
            print("\n💡 Troubleshooting tips:")
            print("- Try webcam index 0: python app.py --single --device 'Test' --webcam 0")
            print("- If you have Mac built-in camera, try: python app.py --single --device 'Test' --webcam 1")
            print("- For RTSP streams: python app.py --single --device 'IP Camera' --webcam 'rtsp://user:pass@ip:port/stream'")
            print("- Check camera permissions in System Preferences > Security & Privacy > Camera")
        finally:
            if processor:
                processor.cleanup()
    
    else:
        if not active_cameras:
            print("❌ No cameras configured! Please uncomment cameras in the CAMERAS list")
            print("\n📋 Available cameras to uncomment:")
            for i, cam in enumerate(CAMERAS):
                print(f"  {i+1}. {cam['device']}: {cam['webcam']}")
            return
        
        print(f"🎯 Starting {len(active_cameras)} camera(s)...")
        print("=" * 50)
        
        for i, camera in enumerate(active_cameras, 1):
            print(f"{i}. {camera['device']}: {camera['webcam']}")
        print("=" * 50)
        
        processes = []
        try:
            for camera in active_cameras:
                process = subprocess.Popen([
                    sys.executable, __file__, 
                    '--single', 
                    '--device', camera['device'], 
                    '--webcam', camera['webcam']
                ])
                processes.append(process)
                print(f"✅ Started {camera['device']} (PID: {process.pid})")
                time.sleep(2)
            
            print(f"\n All {len(processes)} cameras started successfully!")
            print("🛑 Press Ctrl+C to stop all cameras")
            
            for process in processes:
                process.wait()
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping all cameras...")
            
            for process in processes:
                try:
                    process.terminate()
                except Exception as e:
                    print(f"❌ Error terminating process {process.pid}: {e}")
            
            time.sleep(3)
            
            for process in processes:
                try:
                    if process.poll() is None:
                        process.kill()
                        print(f"⚠️ Force killed process {process.pid}")
                    else:
                        print(f"✅ Process {process.pid} stopped gracefully")
                except Exception as e:
                    print(f"❌ Error stopping process {process.pid}: {e}")
            
            print("✅ All cameras stopped.")
        
        except Exception as e:
            print(f"❌ Error starting cameras: {e}")
            for process in processes:
                try:
                    process.terminate()
                except:
                    pass

if __name__ == "__main__":
    main() 