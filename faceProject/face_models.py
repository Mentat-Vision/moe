#!/usr/bin/env python3
"""
Clean face recognition models - only working models included
"""

import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNXRuntime not available")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("⚠️ dlib not available")

import urllib.request
import zipfile
import tempfile

class FaceRecognitionModel:
    """Base class for face recognition models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model_loaded = False
        self.input_size = (112, 112)
        self.embedding_size = 512
        self.model_name = "Base"
    
    def load_model(self):
        """Load the model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def preprocess(self, face_image):
        """Preprocess face image - to be implemented by subclasses"""
        raise NotImplementedError
    
    def extract_embedding(self, face_image):
        """Extract face embedding - to be implemented by subclasses"""
        raise NotImplementedError
    
    def compare_faces(self, embedding1, embedding2):
        """Compare two embeddings using cosine distance"""
        return self.cosine_distance(embedding1, embedding2)
    
    def normalize_embedding(self, embedding):
        """Normalize embedding for cosine distance"""
        return F.normalize(torch.tensor(embedding), p=2, dim=-1).numpy()
    
    def cosine_distance(self, emb1, emb2):
        """Calculate cosine distance between embeddings"""
        emb1 = torch.tensor(emb1, dtype=torch.float32).flatten()
        emb2 = torch.tensor(emb2, dtype=torch.float32).flatten()
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=0)
        emb2 = F.normalize(emb2, p=2, dim=0)
        
        # Calculate cosine similarity
        cosine_sim = torch.dot(emb1, emb2)
        
        # Convert to distance (0 = identical, 2 = opposite)
        return (1 - cosine_sim).item()


class ArcFaceModel(FaceRecognitionModel):
    """ArcFace model using ONNX"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (112, 112)
        self.embedding_size = 512
        self.model_name = "ArcFace"
        self.session = None
        
    def download_model(self):
        """Download ArcFace model if needed"""
        model_path = "models/arcface_resnet100.onnx"
        
        if not os.path.exists(model_path):
            if os.path.exists("models/arcface.onnx"):
                print("Using existing arcface.onnx")
                return True
            
            print(f"Downloading {self.model_name} model...")
            try:
                model_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_path = os.path.join(temp_dir, "buffalo_s.zip")
                    urllib.request.urlretrieve(model_url, zip_path)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        
                        # Find the arcface model file
                        for file in zip_ref.namelist():
                            if 'w600k_r50.onnx' in file:
                                extracted_path = os.path.join(temp_dir, file)
                                if os.path.exists(extracted_path):
                                    os.rename(extracted_path, "models/arcface.onnx")
                                    print(f"✅ Downloaded {self.model_name}")
                                    return True
                                    
                print(f"⚠️ Could not find arcface model in download")
                return False
            except Exception as e:
                print(f"❌ Failed to download {self.model_name}: {e}")
                return False
        return True
        
    def load_model(self):
        """Load ArcFace model"""
        try:
            if not ONNX_AVAILABLE:
                print(f"❌ ONNX Runtime not available for {self.model_name}")
                return False
                
            if not self.download_model():
                return False
                
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            # Try arcface_resnet100.onnx first, then arcface.onnx
            model_file = "models/arcface_resnet100.onnx" if os.path.exists("models/arcface_resnet100.onnx") else "models/arcface.onnx"
            self.session = ort.InferenceSession(model_file, providers=providers)
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
            
        face_resized = cv2.resize(face_image, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
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
    """MobileFaceNet - lightweight face recognition"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (112, 112)
        self.embedding_size = 128
        self.model_name = "MobileFaceNet"
        self.session = None
        
    def load_model(self):
        """Load MobileFaceNet model"""
        try:
            if not ONNX_AVAILABLE:
                print(f"❌ ONNX Runtime not available for {self.model_name}")
                return False
                
            if not os.path.exists("models/mobilefacenet.onnx"):
                print(f"❌ Model file models/mobilefacenet.onnx not found")
                return False
                
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession("models/mobilefacenet.onnx", providers=providers)
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


class DlibResNetModel(FaceRecognitionModel):
    """Dlib ResNet face recognition model"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (150, 150)
        self.embedding_size = 128
        self.model_name = "DlibResNet"
        self.face_rec_model = None
        self.shape_predictor = None
        
    def load_model(self):
        """Load Dlib ResNet model"""
        try:
            if not DLIB_AVAILABLE:
                print(f"❌ dlib not available for {self.model_name}")
                return False
                
            if not os.path.exists("models/shape_predictor_68_face_landmarks.dat"):
                print(f"❌ models/shape_predictor_68_face_landmarks.dat not found")
                return False
                
            if not os.path.exists("models/dlib_face_recognition_resnet_model_v1.dat"):
                print(f"❌ models/dlib_face_recognition_resnet_model_v1.dat not found")
                return False
            
            self.shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
            self.model_loaded = True
            print(f"✅ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def preprocess(self, face_image):
        """Preprocess for Dlib (facial landmarks)"""
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None
            
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def extract_embedding(self, face_image):
        """Extract Dlib ResNet embedding"""
        if not self.model_loaded:
            return None
            
        gray = self.preprocess(face_image)
        if gray is None:
            return None
            
        try:
            # Detect face rectangle
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray)
            
            if len(faces) == 0:
                return None
                
            # Use the first detected face
            face_rect = faces[0]
            
            # Get facial landmarks
            landmarks = self.shape_predictor(gray, face_rect)
            
            # Extract face embedding
            embedding = self.face_rec_model.compute_face_descriptor(gray, landmarks)
            return self.normalize_embedding(np.array(embedding))
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None

class ElasticFaceArcModel(FaceRecognitionModel):
    """ElasticFace with ArcFace loss"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (112, 112) 
        self.embedding_size = 512
        self.model_name = "ElasticFace-Arc"
        self.session = None
        self.model_path = 'models/elasticface_arc.onnx'
        
    def load_model(self):
        """Load ElasticFace-Arc model"""
        try:
            if not ONNX_AVAILABLE:
                print(f"❌ ONNX Runtime not available for {self.model_name}")
                return False
                
            if not os.path.exists(self.model_path):
                print(f"❌ Model file {self.model_path} not found")
                return False
                
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.model_loaded = True
            print(f"✅ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def preprocess(self, face_image):
        """Preprocess for ElasticFace-Arc"""
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None
            
        face_resized = cv2.resize(face_image, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_tensor = face_normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return face_tensor
    
    def extract_embedding(self, face_image):
        """Extract ElasticFace-Arc embedding"""
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

class ElasticFaceCosModel(FaceRecognitionModel):
    """ElasticFace with CosFace loss"""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.input_size = (112, 112)
        self.embedding_size = 512
        self.model_name = "ElasticFace-Cos"
        self.session = None
        self.model_path = 'models/elasticface_cos.onnx'
        
    def load_model(self):
        """Load ElasticFace-Cos model"""
        try:
            if not ONNX_AVAILABLE:
                print(f"❌ ONNX Runtime not available for {self.model_name}")
                return False
                
            if not os.path.exists(self.model_path):
                print(f"❌ Model file {self.model_path} not found")
                return False
                
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.model_loaded = True
            print(f"✅ Loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def preprocess(self, face_image):
        """Preprocess for ElasticFace-Cos"""
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return None
            
        face_resized = cv2.resize(face_image, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_tensor = face_normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return face_tensor
    
    def extract_embedding(self, face_image):
        """Extract ElasticFace-Cos embedding"""
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
        # Only include models that have actual model files available
        self.models = {
            'arcface': ArcFaceModel(device),
            'mobilefacenet': MobileFaceNetModel(device), 
            'dlib_resnet': DlibResNetModel(device),
            'elasticface_arc': ElasticFaceArcModel(device),
            'elasticface_cos': ElasticFaceCosModel(device),
        }
        
        self.current_model = None
        self.current_model_name = None
        
    def list_models(self):
        """List available models"""
        return list(self.models.keys())
    
    def add_model(self, model_name, model_class):
        """Add a model dynamically"""
        if model_name not in self.models:
            self.models[model_name] = model_class(self.device)
            print(f"Added {model_name} to available models")
    
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
        """Compare two embeddings using current model"""
        if self.current_model is None:
            print("❌ No model loaded")
            return float('inf')
        return self.current_model.compare_faces(embedding1, embedding2)
    
    def get_current_model_name(self):
        """Get name of currently loaded model"""
        return self.current_model_name if self.current_model_name else "None"