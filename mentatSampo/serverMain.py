import asyncio
import websockets
import json
import cv2
import numpy as np
import os
import time
import base64
from datetime import datetime

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
        pending_workers = set(self.workers.keys())
        
        async def collect_result(cam_id, worker_name, result):
            """Callback to collect worker results"""
            results[worker_name] = result
            pending_workers.discard(worker_name)
            
            # If all workers have responded, send combined result
            if not pending_workers:
                await self.send_combined_result(websocket, cam_id, results)
        
        # Send frame to all workers
        for worker_name, worker in self.workers.items():
            await worker.add_job(camera_id, frame, collect_result)
        
        # If no workers are available, send empty result
        if not self.workers:
            await websocket.send(json.dumps({
                "camera_id": camera_id,
                "results": {},
                "timestamp": time.time()
            }))
    
    async def route_frame_to_expert(self, camera_id, frame, expert_type, websocket):
        """Route frame to specific expert worker"""
        if expert_type not in self.workers:
            await websocket.send(json.dumps({"error": f"Expert '{expert_type}' not available"}))
            return
        
        # Create callback to send result directly
        async def send_result(cam_id, worker_name, result):
            """Callback to send worker result directly"""
            await websocket.send(json.dumps(result))
        
        # Send frame to specific worker
        worker = self.workers[expert_type]
        await worker.add_job(camera_id, frame, send_result)
    
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
            
        except Exception as e:
            print(f"❌ Error sending result: {e}")
    
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
    
    async def run_server(self):
        """Run the central WebSocket server"""
        # Initialize workers first
        await self.initialize_workers()
        
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