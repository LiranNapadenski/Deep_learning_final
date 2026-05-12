import json
import random
import sys
import os

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def create_trail(dom, noise_pool, is_over_updating, trial_id):
    # We need 3 unique noise blocks for the gaps
    n1, n2, n3 = random.sample(noise_pool, 3)
    
    history = []
    curr_turn = 1

    # STAGE 1: Initial State (Fact A & Dependent Fact B)
    history.append({"turn": curr_turn, "role": "user", "content": dom["fact_a"]})
    curr_turn += 1
    history.append({"turn": curr_turn, "role": "user", "content": dom["fact_b"]})
    curr_turn += 1
    history.append({"turn": curr_turn, "role": "assistant", "content": "I've logged your current situation and plans."})
    curr_turn += 1

    # STAGE 2: Noise Block 1 (8 Turns)
    for turn in n1["turns"]:
        history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
        curr_turn += 1

    if is_over_updating:
        # STAGE 3: Chain of Correction - First Update (Fact A changes)
        history.append({"turn": curr_turn, "role": "user", "content": dom["update_a_v1"]})
        curr_turn += 1
        history.append({"turn": curr_turn, "role": "assistant", "content": "Okay, I've updated the status of that first event."})
        curr_turn += 1

    # STAGE 4: Noise Block 2 (8 Turns)
    # This keeps the total amount of noise and distance comparable between both conditions
    for turn in n2["turns"]:
        history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
        curr_turn += 1

    # STAGE 5: Chain of Correction - Final Negation 
    history.append({"turn": curr_turn, "role": "user", "content": dom["update_a_v2_final"]})
    curr_turn += 1
    history.append({"turn": curr_turn, "role": "assistant", "content": "Got it. I've adjusted your records to the final status."})
    curr_turn += 1

    # STAGE 6: Noise Block 3 (8 Turns)
    for turn in n3["turns"]:
        history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
        curr_turn += 1

    # STAGE 7: Probes (Testing the Ripple)
    history.append({"turn": curr_turn, "role": "user", "content": dom["q_a"]})
    curr_turn += 1
    history.append({"turn": curr_turn, "role": "user", "content": dom["q_b"]})
    curr_turn += 1

    trial = {
        "metadata": {
            "trial_id": f"EXTREME_REVISED_{trial_id:04d}", 
            "domain": dom["domain"],
            "is_over_updating": is_over_updating
        },
        "logic_test": "Chain_Correction + Ripple_Effect",
        "session_history": history,
        "evaluation_probes": [
            {"question": dom["q_a"], "expected": dom["final_a"], "logic": "Chain_Correction_Success"},
            {"question": dom["q_b"], "expected": dom["final_b"], "logic": "Ripple_Effect_Locality"}
        ],
        "is_over_updating": is_over_updating  # Added to the root as requested
    }
    
    return trial

def generate_extreme_revised_dataset(x):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if extreme_domains_revised.json has data, else fallback to extreme_domains.json
    domains_file = os.path.join(base_dir, 'data', 'extreme_domains.json')
    revised_domains_file = os.path.join(base_dir, 'data', 'extreme_domains_revised.json')
    
    if os.path.exists(revised_domains_file) and os.path.getsize(revised_domains_file) > 0:
        print(f"Using revised domains file: {revised_domains_file}")
        domains_file = revised_domains_file
    else:
        print(f"Using default domains file: {domains_file}")

    domains = load_json(domains_file)
    noise_pool = load_json(os.path.join(base_dir, 'data', 'extreme_noise.json'))
    
    final_dataset = []
    trial_counter = 1

    for dom in domains:
        # Check if the domain has the 'over_updating' key, default to True if missing
        is_over_updating = dom.get('over_updating', True)
        
        # Generate x trails for this specific domain configuration
        for _ in range(x):
            trial = create_trail(dom, noise_pool, is_over_updating=is_over_updating, trial_id=trial_counter)
            final_dataset.append(trial)
            trial_counter += 1

    out_file = os.path.join(base_dir, 'extreme_revised_dataset.json')
    with open(out_file, 'w') as f:
        json.dump(final_dataset, f, indent=2)
    print(f"Generated {len(final_dataset)} trials in {out_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            x = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer for x.")
            sys.exit(1)
    else:
        try:
            x = int(input("Enter the number of trials (x) per domain per category: "))
        except ValueError:
            print("Invalid input. Please enter an integer.")
            sys.exit(1)
            
    generate_extreme_revised_dataset(x)
