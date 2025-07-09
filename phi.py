from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

print("🧠 Phi-2 loaded. Type 'exit' to quit.")

chat_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    prompt = chat_history + f"User: {user_input}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
    print("Bot:", response)
    chat_history += f"User: {user_input}\nAssistant: {response}\n"
