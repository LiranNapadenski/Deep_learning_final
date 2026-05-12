import json
import os
import sys

# add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_wrappers.zep_wrapper import ZepWrapper

def test_zep():
    with open("extreme_integrity_dataset.json", "r") as f:
        dataset = json.load(f)
    
    trial = dataset[0]
    zep = ZepWrapper()
    zep.reset()
    
    print("Adding history...")
    zep.add_history(trial['session_history'])
    
    print("Querying A...")
    question_a = trial['evaluation_probes'][0]['question']
    actual_a = zep.query(question_a)
    print("Q:", question_a)
    print("A:", actual_a)

if __name__ == "__main__":
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            if "ZEP_API_KEY" in config:
                os.environ["ZEP_API_KEY"] = config["ZEP_API_KEY"]
            if "GEMINI_API_KEY" in config:
                os.environ["GEMINI_API_KEY"] = config["GEMINI_API_KEY"]
    except Exception as e:
        pass
    test_zep()
