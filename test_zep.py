import json
import os
import time

with open('config.json', 'r') as f:
    config = json.load(f)
    os.environ["ZEP_API_KEY"] = config["ZEP_API_KEY"]
    os.environ["GEMINI_API_KEY"] = config["GEMINI_API_KEY"]

from memory_wrappers.zep_wrapper import ZepWrapper

wrapper = ZepWrapper()
wrapper.reset()
wrapper.add_history([
    {"role": "user", "content": "When do I actually need to be home for the desk delivery? Actually it changed to Wednesday."},
    {"role": "assistant", "content": "Got it. Wednesday for the desk."}
])

ans = wrapper.query("When do I actually need to be home for the desk delivery?")
print("Answer:", ans)
