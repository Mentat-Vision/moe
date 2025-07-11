# llama_server.py
from flask import Flask, request, jsonify
from llama_cpp import Llama
import os

app = Flask(__name__)

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

MODEL_PATH = "./models/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    chat_format="chatml",
    n_ctx=8192,
    n_threads=12,
    n_gpu_layers=35,
    verbose=False
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    history = data.get("history", [])
    user_input = data.get("input", "")

    history.append({"role": "user", "content": user_input})
    response = llm.create_chat_completion(messages=history)
    answer = response["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": answer})

    return jsonify({"response": answer, "history": history})

if __name__ == '__main__':
    # Get server configuration from config.env
    server_ip = config.get("LLAMA_SERVER_IP", "0.0.0.0")
    server_port = int(config.get("LLAMA_SERVER_PORT", 5001))
    
    print(f"🦙 Llama Server starting on {server_ip}:{server_port}")
    app.run(host=server_ip, port=server_port)
