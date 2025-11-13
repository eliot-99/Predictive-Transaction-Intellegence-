import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Replace this with your actual OpenRouter API key
API_KEY = os.getenv('OPENROUTER_API_KEY') 

# Endpoint for OpenRouter
url = "https://openrouter.ai/api/v1/chat/completions"

# Request payload
payload = {
    "model": "deepseek/deepseek-chat",  # Model name
    "messages": [
        {"role": "system", "content": "You are FraudGuard AI, a banking fraud prevention assistant."},
        {"role": "user", "content": "Hello DeepSeek! Can you summarize your capabilities for fraud detection?"}
    ]
}

# Request headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "http://localhost",  # optional but recommended
    "X-Title": "FraudGuard Chatbot Test",      # optional custom name
    "Content-Type": "application/json"
}

# Send the request
response = requests.post(url, headers=headers, json=payload)

# Print response details
if response.status_code == 200:
    data = response.json()
    print("\n✅ Response received successfully!\n")
    print("Model:", data["model"])
    print("Reply:\n", data["choices"][0]["message"]["content"])
else:
    print(f"\n❌ Request failed with status code {response.status_code}")
    print("Response:", response.text)