from llama_cpp import Llama

# Path to your GGUF file
# MODEL_PATH = "./models/llama3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
MODEL_PATH = "./models/llama3/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
SYSTEM_PROMPT = (
    "You are LLaMA-3, an AI assistant. Respond with facts, be concise, and never make up information. "
    "If you don’t know something, say so clearly. Avoid speculation."
)

# === INIT MODEL ===
llm = Llama(
    model_path=MODEL_PATH,
    chat_format="chatml",  # LLaMA 3 uses ChatML
    n_ctx=8192,
    n_threads=12,        # Tune to your CPU
    n_gpu_layers=35,     # Tune to your GPU (set 0 to run CPU-only)
    verbose=False
)

# === INIT HISTORY WITH SYSTEM MESSAGE ===
history = [{"role": "system", "content": SYSTEM_PROMPT}]

print("🦙 LLaMA 3 Chat | type 'exit' to quit.\n")

while True:
    try:
        user_input = input("🔴 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        # Add user message
        history.append({"role": "user", "content": user_input})

        # Call model
        response = llm.create_chat_completion(messages=history)
        answer = response["choices"][0]["message"]["content"]

        # Show response
        print(f"🦙 LLaMA: {answer.strip()}\n")

        # Append model response
        history.append({"role": "assistant", "content": answer.strip()})

    except KeyboardInterrupt:
        print("\n⛔ Exiting.")
        break
    except Exception as e:
        print(f"\n⚠️ Error: {e}\n")
