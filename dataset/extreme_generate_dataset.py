import json
import random

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def generate_extreme_dataset(num_trials=100):
    domains = load_json('data/extreme_domains.json')
    noise_pool = load_json('data/extreme_noise.json')
    final_dataset = []

    for i in range(1, num_trials + 1):
        dom = random.choice(domains)
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

        # STAGE 3: Chain of Correction - First Update (Fact A changes)
        history.append({"turn": curr_turn, "role": "user", "content": dom["update_a_v1"]})
        curr_turn += 1
        history.append({"turn": curr_turn, "role": "assistant", "content": "Okay, I've updated the status of that first event."})
        curr_turn += 1

        # STAGE 4: Noise Block 2 (8 Turns)
        for turn in n2["turns"]:
            history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
            curr_turn += 1

        # STAGE 5: Chain of Correction - Final Negation (Fact A changes again)
        # This is where the Ripple Effect for Fact B is triggered
        neg_turn = curr_turn
        history.append({"turn": curr_turn, "role": "user", "content": dom["update_a_v2_final"]})
        curr_turn += 1
        history.append({"turn": curr_turn, "role": "assistant", "content": "Got it. I've adjusted your records to the final status."})
        curr_turn += 1

        # STAGE 6: Noise Block 3 (8 Turns)
        for turn in n3["turns"]:
            history.append({"turn": curr_turn, "role": turn["role"], "content": turn["content"]})
            curr_turn += 1

        # STAGE 7: Probes (Testing the Ripple)
        probe_a_turn = curr_turn
        history.append({"turn": curr_turn, "role": "user", "content": dom["q_a"]})
        curr_turn += 1
        probe_b_turn = curr_turn
        history.append({"turn": curr_turn, "role": "user", "content": dom["q_b"]})
        curr_turn += 1

        trial = {
            "metadata": {"trial_id": f"EXTREME_{i:03d}", "domain": dom["domain"]},
            "logic_test": "Chain_Correction + Ripple_Effect",
            "session_history": history,
            "evaluation_probes": [
                {"question": dom["q_a"], "expected": dom["final_a"], "logic": "Chain_Correction_Success"},
                {"question": dom["q_b"], "expected": dom["final_b"], "logic": "Ripple_Effect_Locality"}
            ]
        }
        final_dataset.append(trial)

    with open('extreme_integrity_dataset.json', 'w') as f:
        json.dump(final_dataset, f, indent=2)

if __name__ == "__main__":
    generate_extreme_dataset(100)