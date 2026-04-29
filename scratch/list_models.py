import google.generativeai as genai
import os
import json

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
except:
    pass

print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  {m.name}")

print("\nAvailable embedding models:")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"  {m.name}")
