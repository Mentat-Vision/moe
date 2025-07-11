import requests
import json

SERVER_URL = "http://10.8.162.58:5002/chat"

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
