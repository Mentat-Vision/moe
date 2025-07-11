import asyncio
import websockets
import json
import os
import time
from datetime import datetime

def load_config():
    """Load configuration from config.env"""
    config = {
        "yolo_port": 5000,
        "blip_port": 5001
    }
    
    if os.path.exists("config.env"):
        with open("config.env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "YOLO_PORT":
                        config["yolo_port"] = int(value)
                    elif key == "BLIP_PORT":
                        config["blip_port"] = int(value)
    
    return config

class MultiServerManager:
    def __init__(self):
        self.servers = {}
        self.connected_clients = set()
        self.start_time = time.time()
        self.config = load_config()
        
    async def start_yolo_server(self):
        """Start YOLO server"""
        try:
            # Import and start YOLO server
            from experts.serverYolo import YOLOWebSocketServer
            yolo_server = YOLOWebSocketServer()
            await yolo_server.run_server()
        except Exception as e:
            print(f"❌ Error starting YOLO server: {e}")
    
    async def start_blip_server(self):
        """Start BLIP server"""
        try:
            # Import and start BLIP server
            from experts.serverBlip import BLIPWebSocketServer
            blip_server = BLIPWebSocketServer()
            await blip_server.run_server()
        except Exception as e:
            print(f"❌ Error starting BLIP server: {e}")
    
    async def run_servers(self):
        """Run all servers in parallel"""
        print(f"🚀 Starting Multi-Server Manager")
        print(f"🎯 YOLO Server will run on port {self.config['yolo_port']}")
        print(f"📝 BLIP Server will run on port {self.config['blip_port']}")
        
        # Start both servers concurrently
        await asyncio.gather(
            self.start_yolo_server(),
            self.start_blip_server()
        )

async def main():
    manager = MultiServerManager()
    await manager.run_servers()

if __name__ == "__main__":
    asyncio.run(main()) 