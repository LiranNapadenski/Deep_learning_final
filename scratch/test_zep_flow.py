import os
import json
import uuid
from zep_python.client import Zep
from zep_python.types import Message

def test_zep_full_flow():
    config_path = r"d:\Liran's Studies\Master's\Deep_learning_final\config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = config.get("ZEP_API_KEY")
    # For Zep Cloud, sometimes it's better NOT to provide a base_url so the SDK uses its internal default.
    client = Zep(api_key=api_key)
    
    session_id = str(uuid.uuid4())
    print(f"Creating session: {session_id}")
    try:
        res = client.memory.add_session(session_id=session_id)
        print(f"Session created: {res}")
    except Exception as e:
        print(f"Failed to create session: {e}")
        return

    messages = [Message(role="user", content="My name is Liran.")]
    print(f"Adding messages to session: {session_id}")
    try:
        res = client.memory.add(session_id, messages=messages)
        print(f"Messages added: {res}")
    except Exception as e:
        print(f"Failed to add messages: {e}")
        # If it failed here, it might be the 404
        return

    print(f"Searching session: {session_id}")
    try:
        # According to browser, search_sessions is the way.
        # Let's try it with text only first.
        res = client.memory.search_sessions(text="What is my name?")
        print(f"Search results (global): {res}")
        
        # Now try to restrict to session_id
        res = client.memory.search_sessions(text="What is my name?", session_ids=[session_id])
        print(f"Search results (restricted): {res}")
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    test_zep_full_flow()
