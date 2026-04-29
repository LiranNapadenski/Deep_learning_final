import httpx
import json

def test_raw_httpx():
    config_path = r"d:\Liran's Studies\Master's\Deep_learning_final\config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = config.get("ZEP_API_KEY")
    headers = {"Authorization": f"Api-Key {api_key}"}
    
    # Try different base paths
    base_paths = [
        "https://api.getzep.com/api/v2",
        "https://api.getzep.com/v2",
        "https://api.getzep.com/api/v1",
        "https://api.getzep.com/v1",
        "https://api.getzep.com"
    ]
    
    # Try different resource paths
    resources = [
        "/memory/sessions",
        "/sessions",
        "/users"
    ]
    
    for base in base_paths:
        for res in resources:
            url = base + res
            print(f"Testing {url}...")
            try:
                resp = httpx.get(url, headers=headers)
                print(f"Status: {resp.status_code}")
                # print(f"Body: {resp.text[:50]}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    test_raw_httpx()
