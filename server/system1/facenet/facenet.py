#!/usr/bin/env python3
"""
Advanced Face Recognition System
Multi-model support with deletion functionality
"""

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import signal
import sys

# Import our modular face recognition models
try:
    import torch
    from face_models import ModelManager
    FACE_MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ face_models.py not found or PyTorch not available. Using basic face detection only.")
    FACE_MODELS_AVAILABLE = False
    torch = None

class StableFaceTracker:
    def __init__(self):
        # Use optimized cascade for detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize face recognition models
        if FACE_MODELS_AVAILABLE:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_manager = ModelManager(device)
            self.current_model = 'facenet512'  # Default model
            self.use_embeddings = False  # Start with basic tracking, can be enabled
            print(f"🤖 Face recognition models available on {device}")
        else:
            self.model_manager = None
            self.use_embeddings = False
            print("📊 Using basic face tracking only")
        
        # Optimized tracking parameters
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 30  # Longer persistence
        self.max_distance = 100
        self.min_stability_for_ui = 10  # Higher threshold for UI updates
        self.embedding_threshold = 0.6  # Cosine distance threshold for face matching
        
        # Auto-registration parameters
        self.auto_register_threshold = 25  # Auto-register when stability reaches this level
        self.auto_registered_faces = set()  # Track which faces we've auto-registered
        
        # Database
        self.faces_db = {}
        self.load_database()
        
        # Stable faces for UI (only update when very stable)
        self.stable_faces = {}
        self.last_ui_update = 0
        
        os.makedirs('faces', exist_ok=True)
        print(f"Face tracker initialized. Database: {len(self.faces_db)} faces")
        print(f"Auto-registration enabled: faces will be saved automatically with temp IDs")
        
    def detect_faces(self, frame):
        """Optimized face detection"""
        # Resize frame for faster processing
        scale_factor = 0.5
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Fast detection on smaller frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Scale coordinates back to original frame
        face_list = []
        for (x, y, w, h) in faces:
            # Scale back to original coordinates
            x = int(x / scale_factor)
            y = int(y / scale_factor) 
            w = int(w / scale_factor)
            h = int(h / scale_factor)
            
            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_list.append({
                'bbox': [x1, y1, x2, y2],
                'center': [x + w//2, y + h//2],
                'confidence': 1.0,
                'size': w * h
            })
        
        return face_list
    
    def update_tracks(self, faces, frame=None):
        """Stable tracking with face re-identification"""
        current_time = time.time()
        
        if not faces:
            # Just increment disappeared counter
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return list(self.tracks.values())
        
        # Create new tracks for first detections
        if not self.tracks:
            for face in faces:
                # Try to match with existing registered faces first
                matched_id = self.try_match_registered_face(face)
                
                if matched_id:
                    track_id = matched_id
                else:
                    track_id = f"person_{self.next_id:03d}"
                    self.next_id += 1
                
                self.tracks[track_id] = {
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
            return list(self.tracks.values())
        
        # Match faces to existing tracks
        used_tracks = set()
        
        for face in faces:
            best_track = None
            min_distance = float('inf')
            
            # First try to match with existing tracks (position-based)
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                
                dx = face['center'][0] - track['center'][0]
                dy = face['center'][1] - track['center'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_track = track_id
            
            if best_track:
                # Update existing track with smoothing
                track = self.tracks[best_track]
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
                
                used_tracks.add(best_track)
            else:
                # No existing track found - try to match with registered faces
                # First try embedding matching if available (more accurate)
                matched_id = None
                if self.use_embeddings:
                    matched_id = self.try_match_with_embeddings(face, frame)
                
                # Fallback to position/size matching if embedding matching failed
                if not matched_id:
                    matched_id = self.try_match_registered_face(face)
                
                if matched_id and matched_id not in self.tracks:
                    # Reuse the registered face ID
                    track_id = matched_id
                else:
                    # Create completely new track
                    track_id = f"person_{self.next_id:03d}"
                    self.next_id += 1
                
                self.tracks[track_id] = {
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
        
        # Update disappeared counter for unused tracks
        for track_id in self.tracks:
            if track_id not in used_tracks:
                self.tracks[track_id]['disappeared'] += 1
        
        # Auto-register stable faces
        for track_id, track in self.tracks.items():
            if (not track['is_registered'] and 
                track['stability'] >= self.auto_register_threshold and 
                track['disappeared'] == 0 and
                track_id not in self.auto_registered_faces):
                
                # Auto-register this face with a temp ID
                temp_name = f"temp_{track_id}"
                print(f"🤖 Auto-registering stable face: {temp_name}")
                self.auto_registered_faces.add(track_id)
                # We'll register it when we have the current frame
        
        # Remove old tracks, but keep registered faces longer
        for track_id in list(self.tracks.keys()):
            max_disappeared = self.max_disappeared * 3 if self.tracks[track_id]['is_registered'] else self.max_disappeared
            if self.tracks[track_id]['disappeared'] > max_disappeared:
                if track_id in self.auto_registered_faces:
                    self.auto_registered_faces.remove(track_id)
                del self.tracks[track_id]
        
        return list(self.tracks.values())
    
    def try_match_registered_face(self, face):
        """Try to match a detected face with registered faces based on position and size similarity"""
        if not self.faces_db:
            return None
        
        face_center = face['center']
        face_size = face['size']
        
        best_match = None
        best_score = 0
        
        # Look for registered faces that might match
        for face_id, face_data in self.faces_db.items():
            # Use stored center and size if available, otherwise calculate from bbox
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
            # Distance score: closer = better (inverse)
            distance_score = max(0, 1 - (distance / (self.max_distance * 2)))
            # Size score: more similar = better
            size_score = size_ratio
            
            # Combined score
            combined_score = (distance_score * 0.7 + size_score * 0.3)
            
            # If this is a better match than previous best
            if combined_score > best_score and combined_score > 0.4:  # Threshold for matching
                best_score = combined_score
                best_match = face_id
        
        if best_match:
            # Update the stored position for this registered face
            self.faces_db[best_match]['center'] = face_center
            self.faces_db[best_match]['size'] = face_size
            self.faces_db[best_match]['last_seen'] = datetime.now().isoformat()
            
        return best_match
    
    def switch_model(self, model_name):
        """Switch to a different face recognition model"""
        if not FACE_MODELS_AVAILABLE:
            print("❌ Face recognition models not available")
            return False
            
        if self.model_manager.load_model(model_name):
            self.current_model = model_name
            self.use_embeddings = True
            print(f"✅ Switched to {model_name}")
            return True
        else:
            print(f"❌ Failed to switch to {model_name}")
            return False
    
    def get_available_models(self):
        """Get list of available models"""
        if not FACE_MODELS_AVAILABLE:
            return []
        return self.model_manager.list_models()
    
    def delete_face(self, face_id):
        """Delete a registered face"""
        if face_id in self.faces_db:
            # Remove image file
            face_data = self.faces_db[face_id]
            if 'image_path' in face_data and os.path.exists(face_data['image_path']):
                try:
                    os.remove(face_data['image_path'])
                    print(f"🗑️ Deleted image: {face_data['image_path']}")
                except Exception as e:
                    print(f"⚠️ Failed to delete image: {e}")
            
            # Remove from database
            del self.faces_db[face_id]
            
            # Remove from active tracks
            if face_id in self.tracks:
                del self.tracks[face_id]
            
            # Save updated database
            self.save_database()
            print(f"✅ Deleted face: {face_id}")
            return True
        else:
            print(f"❌ Face not found: {face_id}")
            return False
    
    def try_match_with_embeddings(self, face, frame):
        """Try to match face using embeddings (more accurate)"""
        if not self.use_embeddings or not self.model_manager:
            return None
        
        # Extract face region
        x1, y1, x2, y2 = face['bbox']
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
        
        # Extract embedding for current face
        current_embedding = self.model_manager.extract_embedding(face_img)
        if current_embedding is None:
            return None
        
        # Compare with all registered faces
        best_match = None
        best_distance = float('inf')
        
        for face_id, face_data in self.faces_db.items():
            if 'embedding' in face_data:
                distance = self.model_manager.compare_faces(current_embedding, face_data['embedding'])
                
                if distance < best_distance and distance < self.embedding_threshold:
                    best_distance = distance
                    best_match = face_id
        
        if best_match:
            # Update stored embedding with new one (adaptive learning)
            self.faces_db[best_match]['embedding'] = current_embedding
            self.faces_db[best_match]['last_seen'] = datetime.now().isoformat()
            print(f"🎯 Matched face with embedding: {best_match} (distance: {best_distance:.3f})")
        
        return best_match
    
    def benchmark_models(self):
        """Benchmark all available models"""
        if not FACE_MODELS_AVAILABLE:
            print("❌ Face recognition models not available")
            return {}
        
        # Collect test images from registered faces
        test_images = []
        for face_id, face_data in self.faces_db.items():
            if 'image_path' in face_data and os.path.exists(face_data['image_path']):
                img = cv2.imread(face_data['image_path'])
                if img is not None:
                    test_images.append(img)
        
        if not test_images:
            print("❌ No test images available for benchmarking")
            return {}
        
        print(f"🔬 Benchmarking models with {len(test_images)} test images...")
        return self.model_manager.benchmark_models(test_images)
    
    def get_stable_faces_for_ui(self):
        """Get faces stable enough for UI interaction"""
        current_time = time.time()
        
        # Only update stable faces every 2 seconds to prevent flickering
        if current_time - self.last_ui_update < 2.0:
            return list(self.stable_faces.values())
        
        self.last_ui_update = current_time
        new_stable_faces = {}
        
        for track_id, track in self.tracks.items():
            if track['stability'] >= self.min_stability_for_ui and track['disappeared'] == 0:
                new_stable_faces[track_id] = {
                    'id': track_id,
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'name': track['name'],
                    'is_registered': track['is_registered'],
                    'stability': track['stability']
                }
        
        self.stable_faces = new_stable_faces
        return list(self.stable_faces.values())
    
    def save_face_image(self, frame, bbox, face_id):
        """Save face image"""
        x1, y1, x2, y2 = bbox
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size > 0 and face_img.shape[0] > 30 and face_img.shape[1] > 30:
            face_img = cv2.resize(face_img, (150, 150))
            face_path = f'faces/{face_id}.jpg'
            cv2.imwrite(face_path, face_img)
            return face_path
        return None
    
    def load_database(self):
        """Load face database"""
        if os.path.exists('faces_db.json'):
            try:
                with open('faces_db.json', 'r') as f:
                    self.faces_db = json.load(f)
            except:
                self.faces_db = {}
        else:
            self.faces_db = {}
    
    def save_database(self):
        """Save face database"""
        try:
            with open('faces_db.json', 'w') as f:
                json.dump(self.faces_db, f, indent=2)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def register_face(self, frame, bbox, face_id, name=None):
        """Register a face with optional embedding extraction"""
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
            
            # Extract embedding if using face recognition models
            if self.use_embeddings and self.model_manager:
                face_img = frame[y1:y2, x1:x2]
                embedding = self.model_manager.extract_embedding(face_img)
                if embedding is not None:
                    face_data['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
                    print(f"🧠 Extracted {self.current_model} embedding for {name or face_id}")
            
            self.faces_db[face_id] = face_data
            
            # Update track if it exists
            if face_id in self.tracks:
                self.tracks[face_id]['name'] = name or face_id
                self.tracks[face_id]['is_registered'] = True
            
            self.save_database()
            model_info = f" ({self.current_model})" if self.use_embeddings else ""
            print(f"✅ Registered: {name or face_id} ({face_id}){model_info}")
            return True
        return False
    
    def rename_face(self, face_id, new_name):
        """Rename a face"""
        if face_id in self.faces_db:
            old_name = self.faces_db[face_id]['name']
            self.faces_db[face_id]['name'] = new_name
            
            # Update track if it exists
            if face_id in self.tracks:
                self.tracks[face_id]['name'] = new_name
            
            self.save_database()
            print(f"✏️ Renamed: {old_name} → {new_name}")
            return True
        return False

class OptimizedFaceApp:
    def __init__(self):
        self.tracker = StableFaceTracker()
        self.cap = None
        self.running = False
        self.current_frame = None
        self.display_frame = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Processing control
        self.process_every_n_frames = 2  # Process every 2nd frame for speed
        self.frame_counter = 0
        
    def start_camera(self, camera_index=2):
        """Start camera with optimal settings"""
        self.cap = cv2.VideoCapture(camera_index)
        
        # Set optimal resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        print("📹 Camera started: 640x480")
        
    def stop_camera(self):
        """Stop camera"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("⏹️ Camera stopped")
    
    def process_frame(self):
        """Optimized frame processing"""
        if not self.cap or not self.running:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_counter += 1
        
        # Only process face detection every N frames
        if self.frame_counter % self.process_every_n_frames == 0:
            faces = self.tracker.detect_faces(frame)
            tracked_faces = self.tracker.update_tracks(faces, frame)
            
            # Auto-register faces that reached the threshold
            self.auto_register_stable_faces(frame)
        else:
            # Use previous tracking results
            tracked_faces = list(self.tracker.tracks.values())
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Draw overlay
        display_frame = frame.copy()
        self.draw_overlay(display_frame, tracked_faces)
        
        # Store frames
        self.current_frame = frame
        self.display_frame = display_frame
        
        return display_frame
    
    def draw_overlay(self, frame, tracked_faces):
        """Draw clean overlay"""
        h, w = frame.shape[:2]
        
        # Draw faces
        for track in tracked_faces:
            if track['disappeared'] > 0:
                continue
                
            x1, y1, x2, y2 = [int(coord) for coord in track['bbox']]
            name = track['name']
            is_registered = track['is_registered']
            stability = track['stability']
            
            # Color scheme: Red and Black theme
            if is_registered:
                color = (0, 0, 255)  # Red for registered
                thickness = 3
            elif stability > 10:
                color = (0, 140, 255)  # Orange for stable
                thickness = 2
            else:
                color = (100, 100, 100)  # Gray for unstable
                thickness = 1
            
            # Main box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"{name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, scale, 2)
            
            # Black background for text
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), font, scale, color, 2)
        
        # Header - black background
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # Title in red
        title = f"Face Recognition | FPS: {self.fps:.0f} | Faces: {len([t for t in tracked_faces if t['disappeared'] == 0])}"
        cv2.putText(frame, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Controls in white
        controls = "SPACE: Register Stable | R: Register All | Q: Quit | Web: localhost:8080"
        cv2.putText(frame, controls, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def register_stable_faces(self):
        """Register very stable faces only"""
        if not self.current_frame:
            return
        
        count = 0
        for track in self.tracker.tracks.values():
            if (not track['is_registered'] and 
                track['stability'] >= 20 and 
                track['disappeared'] == 0):
                
                success = self.tracker.register_face(
                    self.current_frame, track['bbox'], track['id']
                )
                if success:
                    count += 1
        
        print(f"✅ Registered {count} stable faces")
    
    def register_all_faces(self):
        """Register all visible faces"""
        if not self.current_frame:
            return
        
        count = 0
        for track in self.tracker.tracks.values():
            if not track['is_registered'] and track['disappeared'] == 0:
                success = self.tracker.register_face(
                    self.current_frame, track['bbox'], track['id']
                )
                if success:
                    count += 1
        
        print(f"✅ Registered {count} faces")
    
    def auto_register_stable_faces(self, frame):
        """Auto-register faces that have reached stability threshold"""
        for track_id in list(self.tracker.auto_registered_faces):
            if track_id in self.tracker.tracks:
                track = self.tracker.tracks[track_id]
                if not track['is_registered'] and track['disappeared'] == 0:
                    temp_name = f"temp_{track_id}"
                    success = self.tracker.register_face(
                        frame, track['bbox'], track_id, temp_name
                    )
                    if success:
                        print(f"🤖 Auto-registered: {temp_name}")
                        # Remove from auto-registration set since it's now registered
                        self.tracker.auto_registered_faces.remove(track_id)
    
    def run_background_processing(self):
        """Run background face processing with web display"""
        self.start_camera()
        
        print("\n🎯 Face Recognition System - Web Interface Mode")
        print("📱 Web interface: http://localhost:8080")
        print("\n💡 Features:")
        print("  • Live video feed in web browser")
        print("  • Multiple model support (FaceNet512, ArcFace, MixFaceNets, MobileFaceNet)")
        print("  • Face registration and deletion via web interface")
        print("  • Model switching and benchmarking")
        print("\n⏹️ Press Ctrl+C to stop")
        
        try:
            while self.running:
                frame = self.process_frame()
                # Process frame and generate display frame for web
                time.sleep(0.03)  # ~30 FPS for web streaming
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        
        self.stop_camera()

# Web Interface Handler
class WebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app=None, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
            
        if self.path == '/api/faces':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            faces = self.app.tracker.faces_db
            self.wfile.write(json.dumps(faces).encode())
            
        elif self.path == '/api/current_faces':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get stable faces for UI (reduces flickering)
            stable_faces = self.app.tracker.get_stable_faces_for_ui()
            
            response = {
                'faces': stable_faces,
                'total_registered': len(self.app.tracker.faces_db),
                'active_tracks': len(self.app.tracker.tracks),
                'current_model': self.app.tracker.current_model if self.app.tracker.use_embeddings else 'basic',
                'use_embeddings': self.app.tracker.use_embeddings
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/api/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            models = self.app.tracker.get_available_models()
            response = {
                'available_models': models,
                'current_model': self.app.tracker.current_model,
                'use_embeddings': self.app.tracker.use_embeddings
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/api/video_frame':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if self.app.display_frame is not None:
                # Encode frame as base64 for web display
                import base64
                _, buffer = cv2.imencode('.jpg', self.app.display_frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                response = {
                    'frame': frame_data,
                    'fps': self.app.fps,
                    'timestamp': time.time()
                }
            else:
                response = {
                    'frame': None,
                    'fps': 0,
                    'timestamp': time.time()
                }
            
            self.wfile.write(json.dumps(response).encode())
            
        else:
            # Serve static files (HTML, CSS, JS, images)
            super().do_GET()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        success = False
        
        if self.path == '/api/rename_face':
            face_id = data.get('face_id')
            new_name = data.get('new_name')
            success = self.app.tracker.rename_face(face_id, new_name)
            
        elif self.path == '/api/register_face':
            face_id = data.get('face_id')
            name = data.get('name', face_id)
            
            # Find face in tracks
            face_bbox = None
            for track in self.app.tracker.tracks.values():
                if track['id'] == face_id and track['disappeared'] == 0:
                    face_bbox = track['bbox']
                    break
            
            if face_bbox and self.app.current_frame is not None:
                success = self.app.tracker.register_face(
                    self.app.current_frame, face_bbox, face_id, name
                )
                
        elif self.path == '/api/delete_face':
            face_id = data.get('face_id')
            print(f"🗑️ Delete request for face_id: {face_id}")
            success = self.app.tracker.delete_face(face_id)
            print(f"🗑️ Delete result: {success}")
            
        elif self.path == '/api/switch_model':
            model_name = data.get('model_name')
            success = self.app.tracker.switch_model(model_name)
            
        elif self.path == '/api/benchmark':
            benchmark_results = self.app.tracker.benchmark_models()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'results': benchmark_results}).encode())
            return
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'success': success}).encode())

def run_face_recognition():
    """Main function"""
    app = OptimizedFaceApp()
    
    # Graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down...")
        app.stop_camera()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start web server
    def start_web_server():
        def create_handler(*args, **kwargs):
            return WebHandler(*args, app=app, **kwargs)
        
        server = HTTPServer(('localhost', 8080), create_handler)
        print("🌐 Web interface: http://localhost:8080")
        
        try:
            server.serve_forever()
        except:
            pass
    
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Open browser
    def open_browser():
        time.sleep(1)
        try:
            webbrowser.open('http://localhost:8080')
        except:
            pass
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run background processing
    app.run_background_processing()

if __name__ == "__main__":
    run_face_recognition()