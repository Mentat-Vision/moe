import asyncio
import subprocess
import sys
import signal
import time
from pathlib import Path

class ServerManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
        # Get the directory where this script is located
        self.script_dir = Path(__file__).parent
        
        # Server configurations
        self.servers = [
            {
                "name": "YOLO Server",
                "script": "serverYolo.py",
                "port": 5000,
                "type": "websocket"
            },
            {
                "name": "BLIP Server", 
                "script": "serverBlip.py",
                "port": 5001,
                "type": "http"
            }
        ]
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C to gracefully shutdown all servers"""
        print("\n🛑 Shutting down all servers...")
        self.running = False
        self.stop_all_servers()
        sys.exit(0)
    
    def start_server(self, server_config):
        """Start a single server process"""
        try:
            script_path = self.script_dir / server_config["script"]
            
            # Start the server process
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes.append({
                "process": process,
                "config": server_config
            })
            
            print(f"🚀 Started {server_config['name']} on port {server_config['port']}")
            return process
            
        except Exception as e:
            print(f"❌ Failed to start {server_config['name']}: {e}")
            return None
    
    def start_all_servers(self):
        """Start all servers concurrently"""
        print("🎯 Starting all servers...")
        print("=" * 50)
        
        for server_config in self.servers:
            self.start_server(server_config)
            time.sleep(1)  # Small delay between starts
        
        print("=" * 50)
        print("✅ All servers started!")
        print("📊 Server Status:")
        print(f"   • YOLO WebSocket Server: ws://0.0.0.0:5000")
        print(f"   • BLIP HTTP Server: http://0.0.0.0:5001")
        print("\n🔄 Press Ctrl+C to stop all servers")
        print("-" * 50)
    
    def stop_all_servers(self):
        """Stop all running server processes"""
        for server_info in self.processes:
            process = server_info["process"]
            config = server_info["config"]
            
            try:
                print(f"�� Stopping {config['name']}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {config['name']} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Force killing {config['name']}...")
                process.kill()
            except Exception as e:
                print(f"❌ Error stopping {config['name']}: {e}")
    
    def monitor_servers(self):
        """Monitor server processes and restart if needed"""
        while self.running:
            for server_info in self.processes:
                process = server_info["process"]
                config = server_info["config"]
                
                # Check if process is still running
                if process.poll() is not None:
                    print(f"⚠️  {config['name']} crashed, restarting...")
                    
                    # Remove the dead process
                    self.processes.remove(server_info)
                    
                    # Restart the server
                    new_process = self.start_server(config)
                    if new_process:
                        print(f"✅ {config['name']} restarted")
                    else:
                        print(f"❌ Failed to restart {config['name']}")
            
            time.sleep(2)  # Check every 2 seconds
    
    def run(self):
        """Main run method"""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start all servers
            self.start_all_servers()
            
            # Monitor servers
            self.monitor_servers()
            
        except KeyboardInterrupt:
            print("\n�� Received interrupt signal")
        finally:
            self.stop_all_servers()

def main():
    manager = ServerManager()
    manager.run()

if __name__ == "__main__":
    main() 