import os
import json
from zep_python.client import Zep
from zep_python import ZepEnvironment

def test_final():
    config_path = r"d:\Liran's Studies\Master's\Deep_learning_final\config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = config.get("ZEP_API_KEY")
    # Try with the enum value
    print(f"Testing with ZepEnvironment.DEFAULT.value: {ZepEnvironment.DEFAULT.value}")
    client = Zep(api_key=api_key, base_url=ZepEnvironment.DEFAULT.value)
    try:
        resp = client.memory.list_sessions()
        print(f"Success! {resp}")
    except Exception as e:
        print(f"Error with DEFAULT: {e}")

    # Try with just the base domain
    print("\nTesting with https://api.getzep.com")
    client2 = Zep(api_key=api_key, base_url="https://api.getzep.com")
    try:
        resp = client2.memory.list_sessions()
        print(f"Success! {resp}")
    except Exception as e:
        print(f"Error with base domain: {e}")

if __name__ == "__main__":
    test_final()
