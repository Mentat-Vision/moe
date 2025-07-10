import requests

SERVER_URL = "http://10.8.162.58:5000/chat"

history = []

print("🔁 Chat with LLaMA on Sampo. Type 'exit' to quit.\n")

while True:
    user_input = input("🧍 You: ")
    if user_input.lower().strip() == "exit":
        break

    payload = {
        "input": user_input,
        "history": history
    }

    try:
        response = requests.post(SERVER_URL, json=payload).json()
        reply = response["response"]
        history = response["history"]

        print(f"🤖 LLaMA: {reply}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
