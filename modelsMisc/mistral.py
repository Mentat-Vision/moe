from llama_cpp import Llama

llm = Llama(
    model_path="models/mistral.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=20  # Adjust for your GPU
)

def chat():
    print("🦙 Mistral Chat. Type 'exit' to quit.")
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        prompt = build_prompt(history, user_input)
        output = llm(prompt, max_tokens=512, stop=["</s>", "[INST]"], echo=False)
        response = output["choices"][0]["text"].strip()

        print(f"Bot: {response}")
        history.append((user_input, response))

def build_prompt(history, new_input):
    prompt = ""
    for q, a in history:
        prompt += f"[INST] {q} [/INST] {a} "
    prompt += f"[INST] {new_input} [/INST]"
    return prompt

if __name__ == "__main__":
    chat()
