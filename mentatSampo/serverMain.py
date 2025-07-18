import asyncio
import websockets
import json
import cv2
import numpy as np
import os
import time
import base64
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
from utils.resolution import (
    resize_frame_for_processing, 
    scale_bounding_boxes_for_display,
    scale_bounding_boxes_from_processed_to_display,
    draw_detections_on_frame,
    get_processing_scale_from_config,
    validate_scale_factor
)

def load_config():
    """Load configuration from config.env"""
    config = {}
    
    if os.path.exists("config.env"):
        with open("config.env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    
    return config

# Global AI model controls (affects all cameras)
AI_MODELS = {
    "yolo": {"enabled": True, "name": "YOLO Detection"},
    "blip": {"enabled": True, "name": "BLIP Captioning"},
    # Future models can be added here
}

class CentralWebSocketServer:
    """Central WebSocket server that routes frames to expert workers"""
    
    def __init__(self):
        self.config = load_config()
        self.connected_clients = set()
        
        # Initialize expert workers
        self.workers = {}
        self.results_cache = {}  # Store results per camera
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Web dashboard data
        self.camera_data = {}
        self.camera_frames = {}
        self.latest_results = {}
        
        # Flask app for web dashboard
        self.flask_app = Flask(__name__)
        self.flask_app.config['SECRET_KEY'] = 'mentat_vision_secret_key'
        self.socketio = SocketIO(self.flask_app, cors_allowed_origins="*")
        self.setup_web_routes()
        self.setup_socketio_events()
        
    def setup_web_routes(self):
        """Setup Flask routes for web dashboard"""
        
        @self.flask_app.route('/')
        def dashboard():
            """Main dashboard page"""
            # Get current processing scale from config
            processing_scale = float(self.config.get("PROCESSING_SCALE", 0.5))
            
            return render_template('dashboard.html', 
                                ai_models=AI_MODELS,
                                processing_scale=processing_scale)
        
        @self.flask_app.route('/api/cameras')
        def get_cameras():
            """Get list of available cameras"""
            return jsonify(list(self.camera_data.keys()))
        
        @self.flask_app.route('/api/models')
        def get_models():
            """Get current AI model states"""
            return jsonify({"models": AI_MODELS})
        
        @self.flask_app.route('/api/models/<model_name>/toggle', methods=['POST'])
        def toggle_model(model_name):
            """Toggle AI model on/off globally"""
            if model_name not in AI_MODELS:
                return jsonify({"error": "Model not found"}), 404
            
            try:
                data = request.get_json()
                enabled = data.get('enabled', not AI_MODELS[model_name]['enabled'])
                AI_MODELS[model_name]['enabled'] = enabled
                
                print(f"🔧 {AI_MODELS[model_name]['name']}: {'enabled' if enabled else 'disabled'}")
                
                return jsonify({
                    "success": True, 
                    "model": model_name,
                    "enabled": enabled,
                    "message": f"{AI_MODELS[model_name]['name']} {'enabled' if enabled else 'disabled'}"
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.flask_app.route('/api/camera/<camera_id>/data')
        def get_camera_data(camera_id):
            """Get latest data for specific camera"""
            # Ensure camera_id is string for consistency
            camera_id = str(camera_id)
            if camera_id in self.camera_data:
                data = self.camera_data[camera_id]
                # Only print if there are results
                if data.get('results'):
                    print(f"🔍 API: Camera {camera_id} has {len(data['results'])} expert results")
                return jsonify(data)
            print(f"❌ Camera {camera_id} not found. Available: {list(self.camera_data.keys())}")
            return jsonify({"error": "Camera not found"}), 404
        
        @self.flask_app.route('/api/camera/<camera_id>/stream')
        def camera_stream(camera_id):
            """Stream video frames for specific camera"""
            return Response(
                self.generate_frames(camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.flask_app.route('/video/cam_<camera_id>.mjpg')
        def video_stream_standard(camera_id):
            """Standard video stream endpoint for camera"""
            return Response(
                self.generate_frames(camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.flask_app.route('/api/stats')
        def get_stats():
            """Get server statistics"""
            return jsonify(self.get_server_stats())
        
        @self.flask_app.route('/api/resolution/update', methods=['POST'])
        def update_resolution():
            """Update resolution settings live"""
            try:
                data = request.get_json()
                setting = data.get('setting')
                value = data.get('value')
                
                print(f"🔧 Received resolution update request: {setting} = {value}")
                
                if not setting or value is None:
                    print(f"❌ Missing setting or value: setting={setting}, value={value}")
                    return jsonify({"error": "Missing setting or value"}), 400
                
                # Update the config
                old_value = self.config.get(setting, "not set")
                self.config[setting] = str(value)
                
                print(f"🔧 Updated config: {setting} = {old_value} -> {value}")
                
                # Update config file
                self.update_config_file(setting, str(value))
                
                # Broadcast to connected clients if it's a client setting
                if setting == 'CLIENT_PREVIEW_SCALE':
                    self.broadcast_resolution_update(setting, value)
                
                print(f"🔧 Live resolution update: {setting} = {value}")
                
                return jsonify({
                    "success": True,
                    "setting": setting,
                    "value": value,
                    "message": f"Resolution updated: {setting} = {value}"
                })
                
            except Exception as e:
                print(f"❌ Error updating resolution: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.flask_app.route('/api/resolution/current')
        def get_current_resolution():
            """Get current resolution settings"""
            try:
                return jsonify({
                    "PROCESSING_SCALE": float(self.config.get("PROCESSING_SCALE", 0.5))
                })
            except Exception as e:
                print(f"❌ Error getting resolution settings: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.flask_app.route('/api/camera/<camera_id>/debug')
        def get_camera_debug(camera_id):
            """Debug endpoint to see raw camera data structure"""
            camera_id = str(camera_id)
            if camera_id in self.camera_data:
                return jsonify({
                    "camera_id": camera_id,
                    "raw_data": self.camera_data[camera_id],
                    "available_cameras": list(self.camera_data.keys())
                })
            return jsonify({"error": "Camera not found", "available_cameras": list(self.camera_data.keys())}), 404
    
    def setup_socketio_events(self):
        """Setup SocketIO events for live stats streaming"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            print(f"🔌 SocketIO client connected: {request.sid}")
            emit('connected', {'status': 'Connected to MOE Vision Server'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print(f"🔌 SocketIO client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_camera')
        def handle_subscribe_camera(data):
            """Subscribe to camera stats updates"""
            camera_id = data.get('camera_id')
            if camera_id:
                room = f"camera_{camera_id}"
                join_room(room)
                print(f"📡 Client {request.sid} subscribed to camera {camera_id}")
                emit('subscribed', {'camera_id': camera_id, 'room': room})
        
        @self.socketio.on('unsubscribe_camera')
        def handle_unsubscribe_camera(data):
            """Unsubscribe from camera stats updates"""
            camera_id = data.get('camera_id')
            if camera_id:
                room = f"camera_{camera_id}"
                leave_room(room)
                print(f"📡 Client {request.sid} unsubscribed from camera {camera_id}")
                emit('unsubscribed', {'camera_id': camera_id, 'room': room})
        
        @self.socketio.on('get_all_cameras')
        def handle_get_all_cameras():
            """Get list of all available cameras"""
            emit('cameras_list', {'cameras': list(self.camera_data.keys())})
        
        @self.socketio.on('get_camera_stats')
        def handle_get_camera_stats(data):
            """Get current stats for specific camera"""
            camera_id = data.get('camera_id')
            if camera_id in self.camera_data:
                emit('camera_stats', {
                    'camera_id': camera_id,
                    'data': self.camera_data[camera_id]
                })
            else:
                emit('error', {'message': f'Camera {camera_id} not found'})
    
    def generate_frames(self, camera_id):
        """Generate video frames for web streaming"""
        last_frame_time = 0
        frame_interval = 0.2  # 5 FPS for web streaming (reduced from 10 FPS)

        # Use processing scale for web display
        processing_scale = get_processing_scale_from_config(self.config)

        # Ensure camera_id is string for consistency
        camera_id = str(camera_id)

        while True:
            current_time = time.time()

            if camera_id in self.camera_frames and (current_time - last_frame_time) >= frame_interval:
                frame = self.camera_frames[camera_id].copy()

                # Check if any AI models are enabled
                any_models_enabled = any(AI_MODELS[model]['enabled'] for model in AI_MODELS)

                if any_models_enabled:
                    # Only resize and draw overlays if AI models are enabled
                    frame = resize_frame_for_processing(frame, processing_scale)
                    self.draw_overlays_on_frame(frame, camera_id)
                else:
                    # When no AI models are enabled, just resize for display (faster)
                    # Use a fixed display scale for better performance
                    display_scale = 0.5  # 50% for web display
                    frame = resize_frame_for_processing(frame, display_scale)

                # Encode frame as JPEG with lower quality for better performance
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    last_frame_time = current_time

            time.sleep(0.05)  # Small sleep to prevent busy waiting
    
    def draw_overlays_on_frame(self, frame, camera_id):
        """Draw YOLO detections on frame for web display (no BLIP captions)"""
        # Ensure camera_id is string for consistency
        camera_id = str(camera_id)

        if camera_id not in self.latest_results:
            return

        results = self.latest_results[camera_id]

        # Draw YOLO detections - only if YOLO is enabled globally
        yolo_results = results.get('yolo') or results.get('YOLO')
        if yolo_results and 'detections' in yolo_results and AI_MODELS['yolo']['enabled']:
            detections = yolo_results['detections']
            
            # Get processing scale from config
            processing_scale = get_processing_scale_from_config(self.config)
            
            # Get display frame dimensions
            display_shape = frame.shape
            
            # Scale bounding boxes from processed frame coordinates to display frame coordinates
            scaled_detections = scale_bounding_boxes_from_processed_to_display(
                detections, 
                processing_scale,
                display_shape
            )
            
            # Draw the scaled detections
            draw_detections_on_frame(frame, scaled_detections)
        
        # Removed BLIP caption drawing - captions only show in HTML dashboard

    async def initialize_workers(self):
        """Initialize all expert workers"""
        print("🔧 Initializing expert workers...")
        
        # Initialize YOLO worker
        try:
            from experts.serverYolo import YOLOWorker
            self.workers["yolo"] = YOLOWorker(self.config)
            await self.workers["yolo"].start()
        except Exception as e:
            print(f"❌ Failed to initialize YOLO worker: {e}")
        
        # Initialize BLIP worker
        try:
            from experts.serverBlip import BLIPWorker
            self.workers["blip"] = BLIPWorker(self.config)
            await self.workers["blip"].start()
        except Exception as e:
            print(f"❌ Failed to initialize BLIP worker: {e}")
        
        print(f"✅ Initialized {len(self.workers)} expert workers")
        
        # Print worker status
        for name, worker in self.workers.items():
            stats = worker.get_stats()
            print(f"   🔧 {name.upper()}: FPS={stats['fps']}")

    async def handle_client(self, websocket, path):
        """Handle client WebSocket connection"""
        self.connected_clients.add(websocket)
        client_address = websocket.remote_address
        print(f"🔌 Client connected: {client_address}")
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self.process_frame_message(websocket, message)
                else:
                    # Handle JSON messages (future: commands, status requests)
                    try:
                        data = json.loads(message)
                        await self.handle_json_message(websocket, data)
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({"error": "Invalid JSON message"}))
                        
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {client_address}")
        except Exception as e:
            print(f"❌ Error handling client {client_address}: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def process_frame_message(self, websocket, frame_bytes):
        """Process incoming frame from client (legacy binary protocol)"""
        try:
            # Decode frame
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send(json.dumps({"error": "Invalid frame data"}))
                return
            
            # For now, assume camera_id = 0 (we'll enhance this later)
            camera_id = 0
            
            # Store frame for web dashboard
            self.camera_frames[str(camera_id)] = frame
            
            # Route frame to all enabled workers
            await self.route_frame_to_workers(camera_id, frame, websocket)
            
            self.frame_count += 1
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

    async def process_json_frame_message(self, websocket, data):
        """Process incoming frame from client (new JSON protocol)"""
        try:
            # Extract data from JSON message
            expert_type = data.get("expert")
            camera_id = data.get("camera_id", 0)
            frame_base64 = data.get("frame")
            
            if not expert_type or not frame_base64:
                await websocket.send(json.dumps({"error": "Missing expert type or frame data"}))
                return
            
            # Decode base64 frame
            frame_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send(json.dumps({"error": "Invalid frame data"}))
                return
            
            # Store frame for web dashboard
            self.camera_frames[str(camera_id)] = frame
            
            # Route frame to specific expert worker
            await self.route_frame_to_expert(camera_id, frame, expert_type.lower(), websocket)
            
            self.frame_count += 1
            
        except Exception as e:
            print(f"❌ Error processing JSON frame: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

    async def route_frame_to_workers(self, camera_id, frame, websocket):
        """Route frame to all enabled expert workers"""
        # Create callback to collect results
        results = {}
        enabled_workers = []
        
        # Only include workers for enabled models
        for worker_name, worker in self.workers.items():
            model_key = worker_name.lower()
            if model_key in AI_MODELS and AI_MODELS[model_key]['enabled']:
                enabled_workers.append(worker_name)
        
        pending_workers = set(enabled_workers)
        
        async def collect_result(cam_id, worker_name, result):
            """Callback to collect worker results"""
            results[worker_name] = result
            pending_workers.discard(worker_name)
            
            # If all enabled workers have responded, send combined result
            if not pending_workers:
                await self.send_combined_result(websocket, cam_id, results)
        
        # If no enabled workers, send empty result immediately
        if not enabled_workers:
            await websocket.send(json.dumps({
                "camera_id": camera_id,
                "results": {},
                "timestamp": time.time()
            }))
            return
        
        # Send frame to enabled workers with same processing scale
        scale_factor = get_processing_scale_from_config(self.config)
        processed_frame = resize_frame_for_processing(frame, scale_factor)
        
        for worker_name in enabled_workers:
            worker = self.workers[worker_name]
            await worker.add_job(camera_id, processed_frame, collect_result)

    async def route_frame_to_expert(self, camera_id, frame, expert_type, websocket):
        """Route frame to specific expert worker"""
        if expert_type not in self.workers:
            await websocket.send(json.dumps({"error": f"Expert '{expert_type}' not available"}))
            return
        
        # Get processing scale from config (same for all experts)
        scale_factor = get_processing_scale_from_config(self.config)
        
        # Resize frame for AI processing
        processed_frame = resize_frame_for_processing(frame, scale_factor)
        
        # Create callback to send result directly
        async def send_result(cam_id, worker_name, result):
            """Callback to send worker result directly"""
            await websocket.send(json.dumps(result))
            
            # Store result for web dashboard
            self.update_camera_data(cam_id, worker_name, result)
        
        # Send processed frame to specific worker
        worker = self.workers[expert_type]
        await worker.add_job(camera_id, processed_frame, send_result)

    async def send_combined_result(self, websocket, camera_id, results):
        """Send combined results from all workers to client"""
        try:
            response = {
                "camera_id": camera_id,
                "results": results,
                "timestamp": time.time(),
                "server_stats": self.get_server_stats()
            }
            
            await websocket.send(json.dumps(response))
            
            # Store results for web dashboard
            self.latest_results[str(camera_id)] = results
            for worker_name, result in results.items():
                self.update_camera_data(camera_id, worker_name, result)
            
        except Exception as e:
            print(f"❌ Error sending result: {e}")
    
    def update_camera_data(self, camera_id, worker_name, result):
        """Update camera data for web dashboard with proper structure"""
        # Ensure camera_id is string for consistency
        camera_id = str(camera_id)
        
        # Check if this model is enabled globally
        model_key = worker_name.lower()
        if model_key in AI_MODELS and not AI_MODELS[model_key]['enabled']:
            # Model is disabled, don't update data
            return
        
        if camera_id not in self.camera_data:
            self.camera_data[camera_id] = {
                "timestamp": time.time(),
                "results": {},
                "connected": True
            }
        
        # Store result with proper structure
        self.camera_data[camera_id]["results"][worker_name] = result
        self.camera_data[camera_id]["timestamp"] = time.time()
        self.camera_data[camera_id]["connected"] = True
        
        # Update latest_results for frame overlays
        if camera_id not in self.latest_results:
            self.latest_results[camera_id] = {}
        self.latest_results[camera_id][worker_name] = result
        
        # Debug: print summary of data being stored
        if 'fps' in result:
            print(f"🔍 Camera {camera_id} {worker_name}: FPS={result.get('fps', 'N/A')}")
        
        # Broadcast stats update to SocketIO clients
        self.broadcast_camera_stats(camera_id)

    def update_config_file(self, setting, value):
        """Update config file with new setting"""
        try:
            config_file = "config.env"
            if not os.path.exists(config_file):
                return
            
            # Read current config
            with open(config_file, 'r') as f:
                lines = f.readlines()
            
            # Find and update the setting
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{setting}="):
                    lines[i] = f"{setting}={value}\n"
                    updated = True
                    break
            
            # If setting wasn't found, add it
            if not updated:
                lines.append(f"{setting}={value}\n")
            
            # Write back to file
            with open(config_file, 'w') as f:
                f.writelines(lines)
                
            print(f"💾 Config updated: {setting}={value}")
            
        except Exception as e:
            print(f"❌ Error updating config file: {e}")

    def broadcast_resolution_update(self, setting, value):
        """Broadcast resolution update to connected clients"""
        try:
            update_data = {
                'type': 'resolution_update',
                'setting': setting,
                'value': value,
                'timestamp': time.time()
            }
            
            # Broadcast to all connected clients
            self.socketio.emit('resolution_update', update_data)
            print(f"📡 Broadcasting resolution update: {setting} = {value}")
            
        except Exception as e:
            print(f"❌ Error broadcasting resolution update: {e}")

    def broadcast_camera_stats(self, camera_id):
        """Broadcast camera stats to SocketIO clients subscribed to this camera"""
        try:
            # Ensure camera_id is string for consistency
            camera_id = str(camera_id)
            room = f"camera_{camera_id}"
            
            # Get the camera data
            camera_data = self.camera_data.get(camera_id, {})
            
            # Create properly structured stats data
            stats_data = {
                'camera_id': camera_id,
                'timestamp': camera_data.get("timestamp", time.time()),
                'connected': camera_data.get("connected", False),
                'results': camera_data.get("results", {})
            }
            
            # Debug: Only print if there are actual results
            if stats_data['results']:
                print(f"📡 Broadcasting stats for camera {camera_id}: {list(stats_data['results'].keys())}")
            
            # Emit to both the room and globally for debugging
            self.socketio.emit('camera_stats_update', stats_data, room=room)
            
            # Also emit globally for any clients that might be listening
            self.socketio.emit('camera_stats_update', stats_data)
            
        except Exception as e:
            print(f"❌ Error broadcasting stats for camera {camera_id}: {e}")

    async def handle_json_message(self, websocket, data):
        """Handle JSON command messages from clients"""
        if data.get("type") == "ping":
            await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
        elif data.get("type") == "stats":
            stats = self.get_server_stats()
            await websocket.send(json.dumps({"type": "stats", "data": stats}))
        elif data.get("expert") and data.get("frame"):
            # Handle new protocol: frame processing request
            await self.process_json_frame_message(websocket, data)
        else:
            await websocket.send(json.dumps({"error": "Unknown message type"}))

    def get_server_stats(self):
        """Get server statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        worker_stats = {}
        for name, worker in self.workers.items():
            worker_stats[name] = worker.get_stats()
        
        return {
            "total_frames": self.frame_count,
            "server_fps": round(fps, 2),
            "connected_clients": len(self.connected_clients),
            "workers": worker_stats,
            "uptime": round(elapsed_time, 2)
        }

    def run_flask_app(self):
        """Run Flask app with SocketIO in separate thread"""
        web_host = self.config.get("WEB_HOST", "0.0.0.0")
        web_port = int(self.config.get("WEB_PORT", 5002))
        
        print(f"🌐 Starting web dashboard on http://{web_host}:{web_port}")
        print(f"📡 SocketIO available at ws://{web_host}:{web_port}/socket.io/")
        self.socketio.run(self.flask_app, host=web_host, port=web_port, debug=False)

    async def run_server(self):
        """Run the central WebSocket server"""
        # Initialize workers first
        await self.initialize_workers()
        
        # Start Flask app in separate thread
        flask_thread = threading.Thread(target=self.run_flask_app, daemon=True)
        flask_thread.start()
        
        # Get server configuration
        server_ip = self.config.get("SERVER_IP", "0.0.0.0")
        server_port = int(self.config.get("SERVER_PORT", 5000))
        
        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            server_ip,
            server_port
        )
        
        print(f"🚀 Central WebSocket Server running on {server_ip}:{server_port}")
        print(f"📊 Connected clients: {len(self.connected_clients)}")
        print("💡 Ready to process frames from clients")
        
        await server.wait_closed()

async def main():
    """Main entry point"""
    server = CentralWebSocketServer()
    await server.run_server()

if __name__ == "__main__":
    asyncio.run(main()) 