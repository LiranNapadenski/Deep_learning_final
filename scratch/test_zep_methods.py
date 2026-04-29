import os
import json
from zep_python.client import Zep

def test_zep():
    config_path = r"d:\Liran's Studies\Master's\Deep_learning_final\config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = config.get("ZEP_API_KEY")
    if not api_key:
        print("No ZEP_API_KEY found")
        return

    client = Zep(api_key=api_key)
    print("Methods in client.memory:")
    print(dir(client.memory))

if __name__ == "__main__":
    test_zep()
