# llama_server.py
from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

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
    app.run(host="0.0.0.0", port=5002)
