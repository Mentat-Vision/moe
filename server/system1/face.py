#!/usr/bin/env python3
"""
Stable Face Recognition System for Mentat
Based on faceProject architecture for crash-resistance and stability
Integrated with ArcFace models from faceProject
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
import threading
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import queue

# Import ArcFace dependencies
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Add faceProject to path for model imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'faceProject')))

logger = logging.getLogger(__name__)

class ArcFaceModel:
    """ArcFace model using ONNX for face recognition"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.input_size = (112, 112)
        self.embedding_size = 512
        self.model_name = "ArcFace"
        self.session = None
        self.model_loaded = False
        
    def load_model(self):
        """Load ArcFace ONNX model"""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX Runtime not available")
            return False
            
        # Look for ArcFace model in faceProject models directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Get MOE root directory  
        model_paths = [
            os.path.join(base_dir, "faceProject/models/arcface_resnet100.onnx"),
            os.path.join(base_dir, "faceProject/models/arcface.onnx"),
            os.path.join(base_dir, "faceProject/models/mobilefacenet.onnx"),
            os.path.join(base_dir, "faceProject/models/elasticface_arc.onnx"),
            os.path.join(base_dir, "faceProject/models/elasticface_cos.onnx")
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            logger.warning("ArcFace model not found, using basic face tracking")
            return False
            
        try:
            # Set ONNX providers
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda':
                try:
                    import torch
                    if torch.cuda.is_available():
                        providers.insert(0, 'CUDAExecutionProvider')
                except ImportError:
                    pass
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.model_loaded = True
            logger.info(f"✅ Loaded ArcFace model from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load ArcFace model: {e}")
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
            return self.normalize_embedding(embedding.flatten())
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def normalize_embedding(self, embedding):
        """Normalize embedding for cosine distance"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def cosine_distance(self, emb1, emb2):
        """Calculate cosine distance between embeddings"""
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()
        
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Calculate cosine similarity
        cosine_sim = np.dot(emb1_norm, emb2_norm)
        
        # Convert to distance (0 = identical, 2 = opposite)
        return max(0, 1 - cosine_sim)

class StableFaceTracker:
    """Stable face tracker based on faceProject architecture"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        
        # Initialize face cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.error(f"Failed to load face cascade from {cascade_path}")
                self.face_cascade = None
            else:
                logger.info(f"✅ Loaded face cascade")
        except Exception as e:
            logger.error(f"Error loading face cascade: {e}")
            self.face_cascade = None
        
        # Initialize ArcFace model
        try:
            device = 'cpu'
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                pass
            
            self.arcface_model = ArcFaceModel(device)
            self.use_embeddings = False  # Will be set to True if model loads successfully
            
            # Attempt to load ArcFace model immediately if face recognition is enabled
            if enabled and self.arcface_model.load_model():
                self.use_embeddings = True
                logger.info(f"✅ ArcFace model loaded successfully on {device}")
            else:
                logger.info(f"ArcFace model initialized on {device} (will load on first use)")
        except Exception as e:
            logger.error(f"Failed to initialize ArcFace model: {e}")
            self.arcface_model = None
            self.use_embeddings = False
        
        # Global tracking parameters (based on faceProject)
        self.global_tracks = {}  # Global face tracks across all cameras
        self.next_id = 1
        self.max_disappeared = 30  # Frames before considering face lost
        self.max_distance = 150  # Max distance for face matching
        self.embedding_threshold = 0.6  # Cosine distance threshold for face matching
        
        # Auto-registration parameters (from faceProject)
        self.auto_register_threshold = 25  # Auto-register when stability reaches this level
        self.auto_registered_faces = set()  # Track which faces we've auto-registered
        self.auto_register = True  # Auto-registration enabled by default
        
        # Database - store in system1/faces directory
        self.faces_db = {}
        self.faces_dir = os.path.join(os.path.dirname(__file__), 'faces')
        self.face_database_path = os.path.join(os.path.dirname(__file__), 'faces_db.json')
        os.makedirs(self.faces_dir, exist_ok=True)
        self.load_database()
        
        # Camera management (safe dictionaries)
        self.camera_frames = {}  # Current frames from cameras
        self.camera_queues = defaultdict(lambda: queue.Queue(maxsize=2))
        self.camera_threads = {}
        self.camera_last_process = defaultdict(float)
        self.face_results = defaultdict(lambda: deque(maxlen=20))
        self.annotated_frames = {}  # Store annotated frames with face boxes
        
        # Thread management
        self.running = True
        self.result_callbacks = []
        
        # Processing control
        self.face_interval = 0.3  # Process faces every 300ms
        
        logger.info(f"Face tracker initialized. Database: {len(self.faces_db)} faces")
        logger.info(f"Auto-registration enabled: faces will be saved automatically with temp IDs")
    
    def detect_faces(self, frame, camera_id=None):
        """Optimized face detection based on faceProject"""
        if self.face_cascade is None or frame is None:
            return []
        
        try:
            # Resize frame for faster processing (faceProject pattern)
            scale_factor = 0.75
            small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to improve detection (faceProject pattern)
            gray = cv2.equalizeHist(gray)
            
            # Fast detection on smaller frame - sensitive settings (faceProject pattern)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Handle empty detection results
            if isinstance(faces, tuple) and len(faces) == 0:
                return []
            elif hasattr(faces, 'tolist'):
                face_list = faces.tolist()
            else:
                face_list = list(faces)
            
            # Scale coordinates back to original frame and create face dictionaries
            processed_faces = []
            for (x, y, w, h) in face_list:
                # Scale back to original coordinates
                x = int(x / scale_factor)
                y = int(y / scale_factor) 
                w = int(w / scale_factor)
                h = int(h / scale_factor)
                
                # Add padding (faceProject pattern)
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_dict = {
                    'bbox': [x1, y1, x2, y2],
                    'center': [x + w//2, y + h//2],
                    'confidence': 1.0,
                    'size': w * h
                }
                
                if camera_id:
                    face_dict['camera_id'] = camera_id
                    
                processed_faces.append(face_dict)
            
            if len(processed_faces) > 0:
                logger.debug(f"🔍 Detected {len(processed_faces)} faces on {camera_id or 'unknown camera'}")
            
            return processed_faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def update_tracks(self, faces, frame=None, camera_id=None):
        """Update tracking - based on faceProject architecture"""
        current_time = time.time()
        
        if not faces:
            # Just increment disappeared counter (faceProject pattern)
            for track_id in list(self.global_tracks.keys()):
                self.global_tracks[track_id]['disappeared'] += 1
                if self.global_tracks[track_id]['disappeared'] > self.max_disappeared:
                    if track_id in self.auto_registered_faces:
                        self.auto_registered_faces.remove(track_id)
                    del self.global_tracks[track_id]
            return list(self.global_tracks.values())
        
        # Create new tracks for first detections (faceProject pattern)
        if not self.global_tracks:
            for face in faces:
                # Try to match with existing registered faces first
                matched_id = self.try_match_registered_face(face, frame)
                
                if matched_id:
                    track_id = matched_id
                else:
                    track_id = f"person_{self.next_id:03d}"
                    self.next_id += 1
                
                track_data = {
                    'id': track_id,
                    'center': face['center'],
                    'bbox': face['bbox'],
                    'disappeared': 0,
                    'last_seen': current_time,
                    'stability': 1,
                    'confidence': face['confidence'],
                    'name': self.faces_db.get(track_id, {}).get('name', track_id),
                    'is_registered': track_id in self.faces_db
                }
                
                if camera_id:
                    track_data['camera_id'] = camera_id
                    track_data['camera_history'] = [camera_id]
                
                self.global_tracks[track_id] = track_data
            return list(self.global_tracks.values())
        
        # Match faces to existing tracks (faceProject logic)
        used_tracks = set()
        
        for face in faces:
            best_track = None
            min_distance = float('inf')
            
            # First try to match with existing tracks
            for track_id, track in self.global_tracks.items():
                if track_id in used_tracks:
                    continue
                
                # Calculate distance
                distance = self.calculate_face_distance(face, track)
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_track = track_id
            
            if best_track:
                # Update existing track with smoothing (faceProject pattern)
                track = self.global_tracks[best_track]
                alpha = 0.3  # Less aggressive smoothing
                
                track['center'] = [
                    int(alpha * face['center'][0] + (1-alpha) * track['center'][0]),
                    int(alpha * face['center'][1] + (1-alpha) * track['center'][1])
                ]
                track['bbox'] = face['bbox']
                track['disappeared'] = 0
                track['last_seen'] = current_time
                track['stability'] = min(50, track['stability'] + 1)
                track['confidence'] = face['confidence']
                
                # Update camera info
                if camera_id:
                    if track.get('camera_id') != camera_id:
                        logger.debug(f"🔄 Face {track_id} moved to {camera_id}")
                    track['camera_id'] = camera_id
                    if 'camera_history' not in track:
                        track['camera_history'] = []
                    if camera_id not in track['camera_history']:
                        track['camera_history'].append(camera_id)
                
                used_tracks.add(best_track)
            else:
                # No existing track found - try to match with registered faces
                matched_id = None
                if self.use_embeddings and frame is not None:
                    matched_id = self.try_match_with_embeddings(face, frame)
                
                if not matched_id:
                    matched_id = self.try_match_registered_face(face, frame)
                
                if matched_id and matched_id not in self.global_tracks:
                    # Reuse the registered face ID
                    track_id = matched_id
                else:
                    # Create completely new track
                    track_id = f"person_{self.next_id:03d}"
                    self.next_id += 1
                
                track_data = {
                    'id': track_id,
                    'center': face['center'],
                    'bbox': face['bbox'],
                    'disappeared': 0,
                    'last_seen': current_time,
                    'stability': 1,
                    'confidence': face['confidence'],
                    'name': self.faces_db.get(track_id, {}).get('name', track_id),
                    'is_registered': track_id in self.faces_db
                }
                
                if camera_id:
                    track_data['camera_id'] = camera_id
                    track_data['camera_history'] = [camera_id]
                
                self.global_tracks[track_id] = track_data
        
        # Update disappeared counter for unused tracks
        for track_id in self.global_tracks:
            if track_id not in used_tracks:
                self.global_tracks[track_id]['disappeared'] += 1
        
        # Auto-register stable faces (faceProject pattern)
        for track_id, track in self.global_tracks.items():
            if (not track['is_registered'] and 
                track['stability'] >= self.auto_register_threshold and 
                track['disappeared'] == 0 and
                track_id not in self.auto_registered_faces and
                self.auto_register):
                
                # Find the frame to use for registration
                reg_frame = frame
                if track.get('camera_id') and track['camera_id'] in self.camera_frames:
                    reg_frame = self.camera_frames.get(track['camera_id'], frame)
                
                if reg_frame is not None:
                    temp_name = f"temp_{track_id}"
                    logger.info(f"🤖 Auto-registering stable face: {temp_name}")
                    self.auto_registered_faces.add(track_id)
                    
                    if self.register_face(reg_frame, track['bbox'], track_id, temp_name):
                        track['is_registered'] = True
                        track['name'] = temp_name
        
        # Remove old tracks (faceProject pattern)
        for track_id in list(self.global_tracks.keys()):
            max_disappeared = self.max_disappeared * 3 if self.global_tracks[track_id]['is_registered'] else self.max_disappeared
            if self.global_tracks[track_id]['disappeared'] > max_disappeared:
                if track_id in self.auto_registered_faces:
                    self.auto_registered_faces.remove(track_id)
                del self.global_tracks[track_id]
        
        return list(self.global_tracks.values())
    
    def calculate_face_distance(self, face, track):
        """Calculate distance between face and track"""
        dx = face['center'][0] - track['center'][0]
        dy = face['center'][1] - track['center'][1]
        return np.sqrt(dx*dx + dy*dy)
    
    def try_match_registered_face(self, face, frame):
        """Try to match a detected face with registered faces"""
        if not self.faces_db:
            return None
        
        face_center = face['center']
        face_size = face['size']
        
        best_match = None
        best_score = 0
        
        for face_id, face_data in self.faces_db.items():
            if 'center' in face_data and 'size' in face_data:
                stored_center = face_data['center']
                stored_size = face_data['size']
            elif 'bbox' in face_data:
                stored_bbox = face_data['bbox']
                stored_center = [(stored_bbox[0] + stored_bbox[2]) // 2, (stored_bbox[1] + stored_bbox[3]) // 2]
                stored_size = (stored_bbox[2] - stored_bbox[0]) * (stored_bbox[3] - stored_bbox[1])
            else:
                continue
            
            # Calculate distance
            dx = face_center[0] - stored_center[0]
            dy = face_center[1] - stored_center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Calculate size similarity
            size_ratio = min(face_size, stored_size) / max(face_size, stored_size) if max(face_size, stored_size) > 0 else 0
            
            # Create a matching score (higher is better)
            distance_score = max(0, 1 - (distance / (self.max_distance * 2)))
            size_score = size_ratio
            
            # Combined score
            combined_score = (distance_score * 0.7 + size_score * 0.3)
            
            if combined_score > best_score and combined_score > 0.4:
                best_score = combined_score
                best_match = face_id
        
        if best_match:
            self.faces_db[best_match]['center'] = face_center
            self.faces_db[best_match]['size'] = face_size
            self.faces_db[best_match]['last_seen'] = datetime.now().isoformat()
            
        return best_match
    
    def try_match_with_embeddings(self, face, frame):
        """Try to match face using embeddings (more accurate)"""
        if not self.use_embeddings or not self.arcface_model or not self.arcface_model.model_loaded:
            return None
        
        # Extract face region
        x1, y1, x2, y2 = face['bbox']
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
        
        # Extract embedding for current face
        current_embedding = self.arcface_model.extract_embedding(face_img)
        if current_embedding is None:
            return None
        
        # Compare with all registered faces
        best_match = None
        best_distance = float('inf')
        
        for face_id, face_data in self.faces_db.items():
            if 'embedding' in face_data:
                distance = self.arcface_model.cosine_distance(current_embedding, face_data['embedding'])
                
                if distance < best_distance and distance < self.embedding_threshold:
                    best_distance = distance
                    best_match = face_id
        
        if best_match:
            self.faces_db[best_match]['embedding'] = current_embedding.tolist()
            self.faces_db[best_match]['last_seen'] = datetime.now().isoformat()
            logger.debug(f"🎯 Matched face with embedding: {best_match} (distance: {best_distance:.3f})")
        
        return best_match
    
    def process_faces_in_frame(self, frame: np.ndarray, camera_id: str, timestamp: float) -> List[Dict]:
        """Process all faces in a frame using faceProject patterns"""
        try:
            # Store current frame safely
            self.camera_frames[camera_id] = frame.copy()
            
            # Detect faces
            faces = self.detect_faces(frame, camera_id)
            
            # Update tracking
            tracks = self.update_tracks(faces, frame, camera_id)
            
            # Create results and annotated frame
            results = []
            annotated_frame = frame.copy()
            
            for track in tracks:
                if track['disappeared'] == 0:  # Only show active faces
                    try:
                        # Draw face annotation
                        self.draw_face_annotation(annotated_frame, track['id'], track['name'], track['bbox'], track['is_registered'])
                        
                        # Create result
                        result = {
                            'face_id': track['id'],
                            'name': track['name'],
                            'bbox': track['bbox'],
                            'confidence': track['confidence'],
                            'camera_id': camera_id,
                            'timestamp': timestamp,
                            'stability': track['stability'],
                            'is_registered': track['is_registered']
                        }
                        results.append(result)
                        
                    except Exception as track_error:
                        logger.error(f"Error processing track {track['id']}: {track_error}")
                        continue
            
            # Store annotated frame safely
            if results:
                try:
                    _, encoded_frame = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if encoded_frame is not None:
                        self.annotated_frames[camera_id] = encoded_frame.tobytes()
                except Exception as encode_error:
                    logger.debug(f"Failed to encode annotated frame: {encode_error}")
            else:
                # Clear annotated frame if no faces
                self.annotated_frames.pop(camera_id, None)
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in process_faces_in_frame for {camera_id}: {e}")
            return []
    
    def draw_face_annotation(self, frame, face_id, name, bbox, is_registered):
        """Draw face bounding box and label on frame with faceProject safety"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Clamp coordinates to frame bounds (faceProject pattern)
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Determine color: Green for registered, Red for new faces (requirement)
            if is_registered and not name.startswith('temp_'):
                color = (0, 255, 0)  # Green for saved faces
                thickness = 3
            else:
                color = (0, 0, 255)  # Red for new/temporary faces
                thickness = 2
            
            # Draw main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add corner markers for better visibility (faceProject pattern)
            corner_size = 15
            cv2.line(frame, (x1, y1), (x1 + corner_size, y1), color, thickness + 1)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_size), color, thickness + 1)
            cv2.line(frame, (x2, y1), (x2 - corner_size, y1), color, thickness + 1)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_size), color, thickness + 1)
            cv2.line(frame, (x1, y2), (x1 + corner_size, y2), color, thickness + 1)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_size), color, thickness + 1)
            cv2.line(frame, (x2, y2), (x2 - corner_size, y2), color, thickness + 1)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_size), color, thickness + 1)
            
            # Prepare label text
            status_icon = "●" if is_registered and not name.startswith('temp_') else "○"
            label = f"{status_icon} {name}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            text_thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, scale, text_thickness)
            
            # Enhanced background for text visibility (faceProject pattern)
            padding = 8
            bg_x1 = max(0, x1 - 2)
            bg_y1 = max(0, y1 - text_h - padding * 2)
            bg_x2 = min(w, x1 + text_w + padding * 2)
            bg_y2 = y1
            
            # Semi-transparent black background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # White outline for text visibility
            text_x = x1 + padding
            text_y = y1 - padding
            cv2.putText(frame, label, (text_x, text_y), font, scale, (255, 255, 255), text_thickness + 1)
            cv2.putText(frame, label, (text_x, text_y), font, scale, color, text_thickness)
            
        except Exception as e:
            logger.error(f"Error drawing face annotation: {e}")
    
    def register_face(self, frame, bbox, face_id, name=None):
        """Register a face with optional embedding extraction"""
        try:
            face_path = self.save_face_image(frame, bbox, face_id)
            
            if face_path:
                # Calculate face characteristics for better matching
                x1, y1, x2, y2 = bbox
                center = [(x1 + x2) // 2, (y1 + y2) // 2]
                size = (x2 - x1) * (y2 - y1)
                
                face_data = {
                    'name': name or face_id,
                    'image_path': face_path,
                    'registered_at': datetime.now().isoformat(),
                    'bbox': [int(x) for x in bbox],
                    'center': center,
                    'size': size,
                    'last_seen': datetime.now().isoformat()
                }
                
                # Extract embedding if using ArcFace
                if self.use_embeddings and self.arcface_model and self.arcface_model.model_loaded:
                    face_img = frame[y1:y2, x1:x2]
                    embedding = self.arcface_model.extract_embedding(face_img)
                    if embedding is not None:
                        face_data['embedding'] = embedding.tolist()
                        logger.debug(f"🧠 Extracted ArcFace embedding for {name or face_id}")
                
                self.faces_db[face_id] = face_data
                
                # Update track if it exists
                if face_id in self.global_tracks:
                    self.global_tracks[face_id]['name'] = name or face_id
                    self.global_tracks[face_id]['is_registered'] = True
                
                self.save_database()
                logger.info(f"✅ Registered: {name or face_id} ({face_id})")
                return True
            return False
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False
    
    def save_face_image(self, frame, bbox, face_id):
        """Save face image"""
        try:
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0 and face_img.shape[0] > 30 and face_img.shape[1] > 30:
                face_img = cv2.resize(face_img, (150, 150))
                face_path = os.path.join(self.faces_dir, f'{face_id}.jpg')
                cv2.imwrite(face_path, face_img)
                return f'faces/{face_id}.jpg'  # Relative path for JSON
            return None
        except Exception as e:
            logger.error(f"Error saving face image: {e}")
            return None
    
    def load_database(self):
        """Load face database"""
        try:
            if os.path.exists(self.face_database_path):
                with open(self.face_database_path, 'r') as f:
                    loaded_db = json.load(f)
                
                # Convert embedding lists back to numpy arrays
                self.faces_db = {}
                for face_id, face_data in loaded_db.items():
                    clean_data = face_data.copy()
                    if 'embedding' in clean_data and isinstance(clean_data['embedding'], list):
                        clean_data['embedding'] = np.array(clean_data['embedding'])
                    self.faces_db[face_id] = clean_data
                    
                logger.info(f"Loaded {len(self.faces_db)} faces from database")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.faces_db = {}
    
    def save_database(self):
        """Save face database"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            clean_db = {}
            for face_id, face_data in self.faces_db.items():
                clean_data = face_data.copy()
                if 'embedding' in clean_data and hasattr(clean_data['embedding'], 'tolist'):
                    clean_data['embedding'] = clean_data['embedding'].tolist()
                clean_db[face_id] = clean_data
            
            with open(self.face_database_path, 'w') as f:
                json.dump(clean_db, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_frame(self, camera_id: str, frame_data: bytes, timestamp: Optional[float] = None):
        """Add frame for face processing with faceProject safety patterns"""
        if not self.enabled:
            return
        
        # Validate input
        if not frame_data or len(frame_data) == 0:
            return
            
        if timestamp is None:
            timestamp = time.time()
        
        # Only process if enough time has passed
        last_process = self.camera_last_process[camera_id]
        if timestamp - last_process >= self.face_interval:
            try:
                # Clear old frames from queue to prevent buildup (faceProject pattern)
                while not self.camera_queues[camera_id].empty():
                    try:
                        self.camera_queues[camera_id].get_nowait()
                    except queue.Empty:
                        break
                
                # Add new frame
                try:
                    self.camera_queues[camera_id].put_nowait({
                        'frame_data': frame_data,
                        'timestamp': timestamp,
                        'camera_id': camera_id
                    })
                    self.camera_last_process[camera_id] = timestamp
                except queue.Full:
                    logger.debug(f"Face queue full for camera {camera_id}, skipping frame")
                    return
                
                # Start processing thread if needed (with better error handling)
                if camera_id not in self.camera_threads or not self.camera_threads[camera_id].is_alive():
                    try:
                        # Stop any existing dead thread first
                        if camera_id in self.camera_threads:
                            old_thread = self.camera_threads[camera_id]
                            if not old_thread.is_alive():
                                logger.debug(f"Removing dead thread for {camera_id}")
                                del self.camera_threads[camera_id]
                        
                        self.camera_threads[camera_id] = threading.Thread(
                            target=self._process_camera_queue,
                            args=(camera_id,),
                            daemon=True,
                            name=f"face_proc_{camera_id}"
                        )
                        self.camera_threads[camera_id].start()
                        logger.info(f"Started face processing thread for {camera_id}")
                    except Exception as thread_error:
                        logger.error(f"Failed to start face processing thread for {camera_id}: {thread_error}")
                        # Remove the problematic thread reference
                        self.camera_threads.pop(camera_id, None)
                    
            except Exception as e:
                logger.error(f"Error adding frame for face processing {camera_id}: {e}")
                # Clean up on error
                self.camera_queues[camera_id] = queue.Queue(maxsize=2)
    
    def _process_camera_queue(self, camera_id: str):
        """Process face recognition queue for a specific camera with faceProject safety"""
        logger.info(f"Starting face processing thread for camera {camera_id}")
        
        failed_attempts = 0
        max_failed_attempts = 10  # faceProject pattern
        
        while self.running:
            try:
                # Only process if face recognition is enabled
                if not self.enabled:
                    time.sleep(0.5)
                    failed_attempts = 0  # Reset on disable
                    continue
                
                try:
                    frame_info = self.camera_queues[camera_id].get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Decode frame safely
                try:
                    frame_array = np.frombuffer(frame_info['frame_data'], dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    if frame is None or frame.size == 0:
                        logger.warning(f"Failed to decode frame for camera {camera_id}")
                        failed_attempts += 1
                        if failed_attempts >= max_failed_attempts:
                            logger.error(f"Too many decode failures for {camera_id}, pausing...")
                            time.sleep(5)
                            failed_attempts = 0
                        continue
                    
                    failed_attempts = 0  # Reset on success
                    
                except Exception as decode_error:
                    logger.error(f"Frame decode error for {camera_id}: {decode_error}")
                    failed_attempts += 1
                    if failed_attempts >= max_failed_attempts:
                        time.sleep(5)
                        failed_attempts = 0
                    continue
                
                # Try to load ArcFace model on first use if not loaded
                if self.arcface_model and not self.arcface_model.model_loaded and not self.use_embeddings:
                    try:
                        if self.arcface_model.load_model():
                            self.use_embeddings = True
                            logger.info("ArcFace model loaded successfully on first use")
                    except Exception as model_load_error:
                        logger.debug(f"ArcFace model load failed: {model_load_error}")
                
                # Process faces safely
                try:
                    face_results = self.process_faces_in_frame(
                        frame, camera_id, frame_info['timestamp']
                    )
                    
                    # Store results only if we have faces
                    if face_results:
                        for result in face_results:
                            self.face_results[camera_id].append(result)
                        
                        # Create summary result
                        result_data = {
                            'camera_id': camera_id,
                            'faces': face_results,
                            'timestamp': datetime.fromtimestamp(frame_info['timestamp']).isoformat(),
                            'unix_timestamp': frame_info['timestamp'],
                            'face_count': len(face_results)
                        }
                        
                        # Call callbacks safely
                        for callback in self.result_callbacks:
                            try:
                                callback(result_data)
                            except Exception as callback_error:
                                logger.error(f"Face callback error: {callback_error}")
                    
                except Exception as process_error:
                    logger.error(f"Face processing error for {camera_id}: {process_error}")
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Critical error in face processing thread for {camera_id}: {e}")
                failed_attempts += 1
                if failed_attempts >= max_failed_attempts:
                    logger.error(f"Face processing thread for {camera_id} stopping due to repeated errors")
                    break
                time.sleep(1)  # Longer pause for critical errors
                
        logger.info(f"Face processing thread stopped for camera {camera_id}")
    
    def get_face_database(self) -> Dict:
        """Get the current face database"""
        return dict(self.faces_db)
    
    def get_camera_faces(self, camera_id: str, count: int = 10) -> List[Dict]:
        """Get recent face results for a camera"""
        results = list(self.face_results[camera_id])
        return results[-count:] if results else []
    
    def get_all_camera_faces(self, count: int = 5) -> Dict[str, List[Dict]]:
        """Get recent face results for all cameras"""
        all_results = {}
        for camera_id in self.face_results:
            all_results[camera_id] = self.get_camera_faces(camera_id, count)
        return all_results
    
    def get_annotated_frame(self, camera_id: str) -> Optional[bytes]:
        """Get annotated frame with face boxes for a camera"""
        return self.annotated_frames.get(camera_id)
    
    def register_callback(self, callback):
        """Register callback for face detection results"""
        self.result_callbacks.append(callback)
    
    def get_stats(self) -> Dict:
        """Get face recognition statistics"""
        return {
            'total_registered_faces': len(self.faces_db),
            'active_cameras': len(self.camera_threads),
            'current_model': 'arcface' if self.use_embeddings else 'basic',
            'embeddings_enabled': self.use_embeddings,
            'arcface_available': ONNX_AVAILABLE,
            'auto_register_enabled': self.auto_register
        }
    
    def update_face_name(self, face_id: str, new_name: str) -> bool:
        """Update the name of a registered face"""
        if face_id in self.faces_db:
            self.faces_db[face_id]['name'] = new_name
            
            # Update track if it exists
            if face_id in self.global_tracks:
                self.global_tracks[face_id]['name'] = new_name
            
            self.save_database()
            logger.info(f"Updated face {face_id} name to: {new_name}")
            return True
        return False
    
    def delete_face(self, face_id: str) -> bool:
        """Delete a face from the database"""
        if face_id in self.faces_db:
            # Remove image file
            try:
                face_data = self.faces_db[face_id]
                if 'image_path' in face_data:
                    # Handle both old and new path formats
                    if face_data['image_path'].startswith('faces/'):
                        full_path = os.path.join(os.path.dirname(__file__), face_data['image_path'])
                    else:
                        full_path = os.path.join(self.faces_dir, os.path.basename(face_data['image_path']))
                    if os.path.exists(full_path):
                        os.remove(full_path)
            except Exception as e:
                logger.error(f"Error removing face image: {e}")
            
            # Remove from database
            del self.faces_db[face_id]
            
            # Remove from active tracks
            if face_id in self.global_tracks:
                del self.global_tracks[face_id]
            
            # Remove from auto-registered set
            if face_id in self.auto_registered_faces:
                self.auto_registered_faces.remove(face_id)
            
            self.save_database()
            logger.info(f"Deleted face: {face_id}")
            return True
        else:
            logger.error(f"Face not found: {face_id}")
            return False
    
    def delete_all_faces(self) -> bool:
        """Delete all faces from the database"""
        try:
            # Remove all image files
            for face_id, face_data in self.faces_db.items():
                try:
                    if 'image_path' in face_data:
                        # Handle both old and new path formats
                        if face_data['image_path'].startswith('faces/'):
                            full_path = os.path.join(os.path.dirname(__file__), face_data['image_path'])
                        else:
                            full_path = os.path.join(self.faces_dir, os.path.basename(face_data['image_path']))
                        if os.path.exists(full_path):
                            os.remove(full_path)
                except Exception as e:
                    logger.error(f"Error removing face image for {face_id}: {e}")
            
            # Clear database
            face_count = len(self.faces_db)
            self.faces_db.clear()
            self.global_tracks.clear()
            self.auto_registered_faces.clear()
            self.next_id = 1
            self.save_database()
            
            logger.info(f"Deleted all {face_count} faces from database")
            return True
        except Exception as e:
            logger.error(f"Error deleting all faces: {e}")
            return False
    
    def toggle_auto_register(self) -> bool:
        """Toggle auto-registration of new faces"""
        self.auto_register = not self.auto_register
        logger.info(f"Auto-registration {'enabled' if self.auto_register else 'disabled'}")
        return self.auto_register
    
    def stop(self):
        """Stop all processing threads with faceProject cleanup patterns"""
        logger.info("Stopping face recognition system...")
        self.running = False
        
        # Stop all camera threads gracefully
        active_threads = []
        for camera_id, thread in self.camera_threads.items():
            if thread and thread.is_alive():
                active_threads.append((camera_id, thread))
        
        if active_threads:
            logger.info(f"Stopping {len(active_threads)} active face processing threads...")
            for camera_id, thread in active_threads:
                try:
                    thread.join(timeout=3.0)
                    if thread.is_alive():
                        logger.warning(f"Face processing thread for {camera_id} did not stop gracefully")
                    else:
                        logger.debug(f"Face processing thread for {camera_id} stopped")
                except Exception as e:
                    logger.error(f"Error stopping face processing thread for {camera_id}: {e}")
        
        # Clear all data structures
        try:
            self.camera_queues.clear()
            self.camera_threads.clear()
            self.annotated_frames.clear()
            self.face_results.clear()
            self.camera_frames.clear()
        except Exception as e:
            logger.error(f"Error clearing face recognition data: {e}")
            
        logger.info("Face recognition stopped")

# Global face recognizer instance
face_recognizer = None

def initialize_face_recognizer(enabled: bool = False):
    """Initialize the global face recognizer"""
    global face_recognizer
    
    try:
        # Stop existing recognizer if any
        if face_recognizer is not None:
            try:
                face_recognizer.stop()
            except Exception as e:
                logger.warning(f"Error stopping existing face recognizer: {e}")
        
        # Create new recognizer based on faceProject architecture
        face_recognizer = StableFaceTracker(enabled=enabled)
        
        logger.info(f"Stable face recognizer initialized (enabled={enabled})")
        
        return face_recognizer
        
    except Exception as e:
        logger.error(f"Failed to initialize face recognizer: {e}")
        # Create a minimal fallback recognizer
        face_recognizer = StableFaceTracker(enabled=False)
        return face_recognizer

def get_face_recognizer():
    """Get the global face recognizer instance"""
    global face_recognizer
    if face_recognizer is None:
        face_recognizer = initialize_face_recognizer()
    return face_recognizer

def process_frame(camera_id: str, frame_data: bytes, timestamp: Optional[float] = None):
    """Process a frame for face recognition"""
    recognizer = get_face_recognizer()
    recognizer.add_frame(camera_id, frame_data, timestamp)

def get_face_database():
    """Get the face database"""
    recognizer = get_face_recognizer()
    return recognizer.get_face_database()

def update_face_name(face_id: str, new_name: str):
    """Update a face name"""
    recognizer = get_face_recognizer()
    return recognizer.update_face_name(face_id, new_name)

def delete_face(face_id: str):
    """Delete a face"""
    recognizer = get_face_recognizer()
    return recognizer.delete_face(face_id)

def delete_all_faces():
    """Delete all faces"""
    recognizer = get_face_recognizer()
    return recognizer.delete_all_faces()

def toggle_auto_register():
    """Toggle auto-registration"""
    recognizer = get_face_recognizer()
    return recognizer.toggle_auto_register()

def get_camera_faces(camera_id: str, count: int = 10):
    """Get faces for a specific camera"""
    recognizer = get_face_recognizer()
    return recognizer.get_camera_faces(camera_id, count)

def get_all_camera_faces(count: int = 5):
    """Get faces for all cameras"""
    recognizer = get_face_recognizer()
    return recognizer.get_all_camera_faces(count)

def get_face_stats():
    """Get face recognition statistics"""
    recognizer = get_face_recognizer()
    return recognizer.get_stats()

def get_face_annotated_frame(camera_id: str):
    """Get annotated frame with face boxes"""
    recognizer = get_face_recognizer()
    return recognizer.get_annotated_frame(camera_id)

if __name__ == "__main__":
    # Test the face recognizer
    recognizer = initialize_face_recognizer()
    print("Stable face recognizer initialized")
    print(f"Stats: {recognizer.get_stats()}")