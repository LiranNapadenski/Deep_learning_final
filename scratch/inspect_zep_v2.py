import os
import json
from zep_python.client import Zep

def inspect_zep():
    config_path = r"d:\Liran's Studies\Master's\Deep_learning_final\config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = config.get("ZEP_API_KEY")
    if not api_key:
        print("No ZEP_API_KEY found")
        return

    client = Zep(api_key=api_key)
    print("Top-level attributes in Zep client:")
    print(dir(client))
    
    if hasattr(client, 'memory'):
        print("\nAttributes in client.memory:")
        print(dir(client.memory))
    
    if hasattr(client, 'graph'):
        print("\nAttributes in client.graph:")
        print(dir(client.graph))

if __name__ == "__main__":
    inspect_zep()
