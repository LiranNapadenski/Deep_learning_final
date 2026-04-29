import os
import uuid
import time
import json
from zep_cloud.client import Zep
from zep_cloud import Message

def test_graph_extraction():
    config_path = r"d:\Liran's Studies\Master's\Deep_learning_final\config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = config.get("ZEP_API_KEY")
    client = Zep(api_key=api_key)
    
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    thread_id = f"test_thread_{uuid.uuid4().hex[:8]}"
    
    print(f"Creating user {user_id} and thread {thread_id}...")
    client.user.add(user_id=user_id)
    client.thread.create(user_id=user_id, thread_id=thread_id)
    
    messages = [
        Message(role="user", content="My cat's name is Luna."),
        Message(role="assistant", content="Got it, Luna."),
        Message(role="user", content="Luna is 3 years old.")
    ]
    
    print("Adding messages...")
    client.thread.add_messages(thread_id, messages=messages)
    
    print("Polling for facts...")
    start = time.time()
    while time.time() - start < 60:
        res = client.graph.search(user_id=user_id, query="Luna")
        if res.edges:
            print(f"Found facts after {time.time() - start:.1f}s!")
            for e in res.edges:
                print(f"- {e.fact}")
            return
        time.sleep(5)
    print("Timed out waiting for facts.")

if __name__ == "__main__":
    test_graph_extraction()
