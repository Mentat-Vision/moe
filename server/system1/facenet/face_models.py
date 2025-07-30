#!/usr/bin/env python3
"""
Modular Face Recognition Models
Supports FaceNet512, ArcFace, MixFaceNets, MobileFaceNet
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import onnxruntime as ort
from abc import ABC, abstractmethod
import os
import urllib.request
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class FaceRecognitionModel(ABC):
    """Abstract base class for face recognition models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.input_size = (112, 112)  # Default
        self.embedding_size = 512  # Default
        self.model_loaded = False
        
    @abstractmethod
    def load_model(self):
        """Load the face recognition model"""
        pass
    
    @abstractmethod
    def preprocess(self, face_image):
        """Preprocess face image for the model"""
        pass
    
    @abstractmethod
    def extract_embedding(self, face_image):
        """Extract face embedding"""
        pass
    
    def normalize_embedding(self, embedding):
        """Normalize embedding for cosine distance"""
        return F.normalize(torch.tensor(embedding), p=2, dim=-1).numpy()
    
    def cosine_distance(self, emb1, emb2):
        """Calculate cosine distance between embeddings"""
        emb1 = torch.tensor(emb1).flatten()
        emb2 = torch.tensor(emb2).flatten()
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=0)
        emb2 = F.normalize(emb2, p=2, dim=0)
        
        # Calculate cosine similarity
        cosine_sim = torch.dot(emb1, emb2)
        
        # Convert to distance (0 = identical, 2 = opposite)
        return (1 - cosine_sim).item()

class FaceNet512Model(FaceRecognitionModel):
    """FaceNet with 512-dimensional embeddings"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (160, 160)
        self.embedding_size = 512
        self.model_name = "FaceNet512"
        
    def load_model(self):
        """Load FaceNet model"""
        try:
            from facenet_pytorch import InceptionResnetV1
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model_loaded = True
            print(f"✅ Loaded {self.model_name}")
            return True
        except ImportError:
            print(f"❌ Failed to load {self.model_name}: facenet-pytorch not installed")
            return False
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def preprocess(self, face_image):
        """Preprocess for FaceNet (160x160, normalized)"""
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None
            
        # Resize to 160x160
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        
        # Convert to tensor
        face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1)).unsqueeze(0)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            face_tensor = face_tensor.cuda()
            
        return face_tensor
    
    def extract_embedding(self, face_image):
        """Extract 512-dim embedding"""
        if not self.model_loaded:
            return None
            
        face_tensor = self.preprocess(face_image)
        if face_tensor is None:
            return None
            
        try:
            with torch.no_grad():
                embedding = self.model(face_tensor)
                return self.normalize_embedding(embedding.cpu().numpy())
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

class ArcFaceModel(FaceRecognitionModel):
    """ArcFace model using ONNX"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (112, 112)
        self.embedding_size = 512
        self.model_name = "ArcFace"
        self.session = None
        
    def download_model(self):
        """Download ArcFace ONNX model"""
        model_path = "arcface_resnet100.onnx"
        
        if not os.path.exists(model_path):
            print(f"Downloading {self.model_name} model...")
            try:
                if HF_AVAILABLE:
                    # Download from HuggingFace
                    downloaded_path = hf_hub_download(
                        repo_id="FoivosPar/Arc2Face", 
                        filename="arcface.onnx",
                        local_dir="."
                    )
                    # Rename to expected filename
                    if os.path.exists("arcface.onnx"):
                        os.rename("arcface.onnx", model_path)
                    print(f"✅ Downloaded {model_path} from HuggingFace")
                else:
                    print(f"❌ HuggingFace Hub not available. Install with: pip install huggingface_hub")
                    return False
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                return False
        return True
        
    def load_model(self):
        """Load ArcFace ONNX model"""
        try:
            if not self.download_model():
                return False
                
            # Set ONNX providers
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession("arcface_resnet100.onnx", providers=providers)
            self.model_loaded = True
            print(f"✅ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def preprocess(self, face_image):
        """Preprocess for ArcFace (112x112, normalized)"""
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None
            
        # Resize to 112x112
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW and add batch dimension
        face_tensor = face_normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return face_tensor
    
    def extract_embedding(self, face_image):
        """Extract ArcFace embedding"""
        if not self.model_loaded:
            return None
            
        face_tensor = self.preprocess(face_image)
        if face_tensor is None:
            return None
            
        try:
            input_name = self.session.get_inputs()[0].name
            embedding = self.session.run(None, {input_name: face_tensor})[0]
            return self.normalize_embedding(embedding)
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

class MobileFaceNetModel(FaceRecognitionModel):
    """MobileFaceNet - lightweight model"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (112, 112)
        self.embedding_size = 128
        self.model_name = "MobileFaceNet"
        self.session = None
        
    def download_model(self):
        """Download MobileFaceNet ONNX model"""
        model_path = "mobilefacenet.onnx"
        
        if not os.path.exists(model_path):
            print(f"⚠️ {self.model_name} model not available for auto-download.")
            print(f"To use MobileFaceNet:")
            print(f"  1. Clone: git clone https://github.com/foamliu/MobileFaceNet.git")
            print(f"  2. Download weights and convert to ONNX following their instructions")
            print(f"  3. Copy the generated ONNX file to {model_path}")
            return False
        return True
        
    def load_model(self):
        """Load MobileFaceNet ONNX model"""
        try:
            if not self.download_model():
                return False
                
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession("mobilefacenet.onnx", providers=providers)
            self.model_loaded = True
            print(f"✅ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def preprocess(self, face_image):
        """Preprocess for MobileFaceNet"""
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None
            
        face_resized = cv2.resize(face_image, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_tensor = face_normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return face_tensor
    
    def extract_embedding(self, face_image):
        """Extract MobileFaceNet embedding"""
        if not self.model_loaded:
            return None
            
        face_tensor = self.preprocess(face_image)
        if face_tensor is None:
            return None
            
        try:
            input_name = self.session.get_inputs()[0].name
            embedding = self.session.run(None, {input_name: face_tensor})[0]
            return self.normalize_embedding(embedding)
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

class MixFaceNetsModel(FaceRecognitionModel):
    """MixFaceNets ensemble model"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (112, 112)
        self.embedding_size = 512
        self.model_name = "MixFaceNets"
        self.session = None
        
    def download_model(self):
        """Download MixFaceNets model"""
        model_path = "mixfacenets.onnx"
        
        if not os.path.exists(model_path):
            print(f"⚠️ {self.model_name} model not available for auto-download.")
            print(f"Please manually download a MixFaceNets ONNX model to {model_path}")
            return False
        return True
        
    def load_model(self):
        """Load MixFaceNets model"""
        try:
            if not self.download_model():
                return False
                
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession("mixfacenets.onnx", providers=providers)
            self.model_loaded = True
            print(f"✅ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def preprocess(self, face_image):
        """Preprocess for MixFaceNets"""
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None
            
        face_resized = cv2.resize(face_image, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        face_tensor = face_normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return face_tensor
    
    def extract_embedding(self, face_image):
        """Extract MixFaceNets embedding"""
        if not self.model_loaded:
            return None
            
        face_tensor = self.preprocess(face_image)
        if face_tensor is None:
            return None
            
        try:
            input_name = self.session.get_inputs()[0].name
            embedding = self.session.run(None, {input_name: face_tensor})[0]
            return self.normalize_embedding(embedding)
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

class ModelManager:
    """Manages multiple face recognition models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {
            'facenet512': FaceNet512Model(device),
            'arcface': ArcFaceModel(device),
            'mobilefacenet': MobileFaceNetModel(device),
            'mixfacenets': MixFaceNetsModel(device)
        }
        self.current_model = None
        self.current_model_name = None
        
    def list_models(self):
        """List available models"""
        return list(self.models.keys())
    
    def load_model(self, model_name):
        """Load a specific model"""
        if model_name not in self.models:
            print(f"❌ Unknown model: {model_name}")
            return False
            
        print(f"Loading {model_name}...")
        success = self.models[model_name].load_model()
        
        if success:
            self.current_model = self.models[model_name]
            self.current_model_name = model_name
            print(f"✅ Switched to {model_name}")
            return True
        else:
            print(f"❌ Failed to load {model_name}")
            return False
    
    def extract_embedding(self, face_image):
        """Extract embedding using current model"""
        if self.current_model is None:
            print("❌ No model loaded")
            return None
            
        return self.current_model.extract_embedding(face_image)
    
    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings"""
        if self.current_model is None:
            return 1.0  # Max distance
            
        return self.current_model.cosine_distance(embedding1, embedding2)
    
    def get_model_info(self):
        """Get current model information"""
        if self.current_model is None:
            return {"name": "None", "input_size": None, "embedding_size": None}
            
        return {
            "name": self.current_model_name,
            "input_size": self.current_model.input_size,
            "embedding_size": self.current_model.embedding_size
        }
    
    def benchmark_models(self, test_images):
        """Benchmark all available models"""
        results = {}
        
        for model_name in self.models.keys():
            print(f"\n🔄 Benchmarking {model_name}...")
            
            if not self.load_model(model_name):
                results[model_name] = {"status": "failed", "error": "Failed to load"}
                continue
            
            import time
            start_time = time.time()
            embeddings_extracted = 0
            
            for img in test_images:
                embedding = self.extract_embedding(img)
                if embedding is not None:
                    embeddings_extracted += 1
            
            end_time = time.time()
            
            results[model_name] = {
                "status": "success",
                "embeddings_extracted": embeddings_extracted,
                "total_images": len(test_images),
                "time_taken": end_time - start_time,
                "fps": embeddings_extracted / (end_time - start_time) if end_time > start_time else 0,
                "model_info": self.get_model_info()
            }
            
            print(f"✅ {model_name}: {embeddings_extracted}/{len(test_images)} embeddings, {results[model_name]['fps']:.2f} FPS")
        
        return results

if __name__ == "__main__":
    # Test the model manager
    manager = ModelManager()
    
    print("Available models:", manager.list_models())
    
    # Try to load FaceNet512
    if manager.load_model('facenet512'):
        print("Model loaded successfully!")
        print("Model info:", manager.get_model_info())
    else:
        print("Failed to load model")