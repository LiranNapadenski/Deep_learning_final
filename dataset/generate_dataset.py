import json
import random

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def generate_staggered_dataset(num_trials=100):
    domains = load_json('dataset/data/domains.json')
    noise_pool = load_json('dataset/data/noise.json')
    final_dataset = []

    for i in range(1, num_trials + 1):
        # Select 1 Domain and 3 unique 8-turn Noise Modules
        dom = random.choice(domains)
        n1, n2, n3 = random.sample(noise_pool, 3)
        
        history = []
        curr_turn = 1

        # 1. Inject Fact A
        history.append({"turn": curr_turn, "role": "user", "content": dom["fact_a"]})
        curr_turn += 1
        history.append({"turn": curr_turn, "role": "assistant", "content": "I've noted that. Anything else?"})
        curr_turn += 1

        # 2. Noise Block 1 (8 Turns of Interference)
        for turn in n1["turns"]:
            history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
            curr_turn += 1

        # 3. Inject Fact B (Now separated from A by 8+ turns)
        fact_b_turn = curr_turn
        history.append({"turn": curr_turn, "role": "user", "content": dom["fact_b"]})
        curr_turn += 1
        history.append({"turn": curr_turn, "role": "assistant", "content": "Understood, I've added that to the record."})
        curr_turn += 1

        # 4. Noise Block 2 (8 Turns of Consolidation)
        for turn in n2["turns"]:
            history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
            curr_turn += 1

        # 5. Negation/Update of Fact A
        neg_turn = curr_turn
        history.append({"turn": curr_turn, "role": "user", "content": dom["update_a"]})
        curr_turn += 1
        history.append({"turn": curr_turn, "role": "assistant", "content": "I have updated my information regarding that first point."})
        curr_turn += 1

        # 6. Noise Block 3 (8 Turns of Final Interference)
        for turn in n3["turns"]:
            history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
            curr_turn += 1

        # 7. Probes
        probe_a_turn = curr_turn
        history.append({"turn": curr_turn, "role": "user", "content": dom["q_a"]})
        curr_turn += 1
        probe_b_turn = curr_turn
        history.append({"turn": curr_turn, "role": "user", "content": dom["q_b"]})
        curr_turn += 1

        # Assemble Trial
        trial = {
            "metadata": {
                "trial_id": f"STAGGERED_{i:03d}",
                "domain": dom["domain"],
                "is_negation": dom["is_negation"]
            },
            "episodic_claims": {
                "claim_A": {
                    "original": dom["fact_a"],
                    "update": dom["update_a"],
                    "status": "Negated" if dom["is_negation"] else "Consistent",
                    "final_truth": dom["final_a"]
                },
                "claim_B": {
                    "statement": dom["fact_b"],
                    "status": "Stable",
                    "final_truth": dom["final_b"]
                }
            },
            "session_map": {
                "fact_a_turn": 1,
                "fact_b_turn": fact_b_turn,
                "negation_turn": neg_turn,
                "probe_turns": [probe_a_turn, probe_b_turn],
                "noise_sequence": [n1["module_name"], n2["module_name"], n3["module_name"]]
            },
            "session_history": history,
            "evaluation_probes": [
                {"question": dom["q_a"], "expected": dom["final_a"], "logic": "Update_Success"},
                {"question": dom["q_b"], "expected": dom["final_b"], "logic": "Locality_Integrity"}
            ]
        }
        final_dataset.append(trial)

    with open('conflict_interference_dataset.json', 'w') as f:
        json.dump(final_dataset, f, indent=2)
    print(f"Successfully generated {num_trials} staggered trials.")

if __name__ == "__main__":
    generate_staggered_dataset(100)