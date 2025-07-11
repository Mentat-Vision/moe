import requests
import json
import os

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

# Load configuration
config = load_config()

# Get server URL from config
server_ip = config.get("LLAMA_SERVER_IP", "10.8.162.58")
server_port = config.get("LLAMA_SERVER_PORT", "5001")
SERVER_URL = f"http://{server_ip}:{server_port}/chat"

def send_message(message, history=None):
    if history is None:
        history = []
    
    data = {
        "input": message,
        "history": history
    }
    
    try:
        response = requests.post(SERVER_URL, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def main():
    print("🤖 Llama Chat Client")
    print(f"Connected to: {SERVER_URL}")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    history = []
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
            
        result = send_message(user_input, history)
        
        if result:
            print(f"Assistant: {result['response']}")
            history = result['history']
        else:
            print("❌ Failed to get response")

if __name__ == "__main__":
    main()
