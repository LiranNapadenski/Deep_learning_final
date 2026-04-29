import json
import os

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def format_turn(turn):
    """Formats a turn dictionary into a single string for MAB ingestion."""
    return f"{turn['role'].capitalize()}: {turn['content']}"

def convert_to_mab(source_data):
    mab_dataset = []

    for trial in source_data:
        history = trial['session_history']
        s_map = trial['session_map']
        
        # MAB expects 'context_chunks'. We group your staggered turns into 
        # logical blocks to preserve the 8-turn gap experiment.
        chunks = []
        
        # Chunk 1: Initial Injection (Fact A)
        chunk_1 = [format_turn(t) for t in history if t['turn'] <= 2]
        chunks.append("\n".join(chunk_1))
        
        # Chunk 2: Noise Block 1
        chunk_2 = [format_turn(t) for t in history if 2 < t['turn'] < s_map['fact_b_turn']]
        chunks.append("\n".join(chunk_2))
        
        # Chunk 3: Fact B Injection
        chunk_3 = [format_turn(t) for t in history if s_map['fact_b_turn'] <= t['turn'] < s_map['fact_b_turn'] + 2]
        chunks.append("\n".join(chunk_3))
        
        # Chunk 4: Noise Block 2
        chunk_4 = [format_turn(t) for t in history if s_map['fact_b_turn'] + 2 <= t['turn'] < s_map['negation_turn']]
        chunks.append("\n".join(chunk_4))
        
        # Chunk 5: The Negation/Update
        chunk_5 = [format_turn(t) for t in history if s_map['negation_turn'] <= t['turn'] < s_map['negation_turn'] + 2]
        chunks.append("\n".join(chunk_5))
        
        # Chunk 6: Noise Block 3
        chunk_6 = [format_turn(t) for t in history if s_map['negation_turn'] + 2 <= t['turn'] < s_map['probe_turns'][0]]
        chunks.append("\n".join(chunk_6))

        # Format Queries for MAB
        mab_queries = []
        for i, probe in enumerate(trial['evaluation_probes']):
            mab_queries.append({
                "qa_pair_id": f"{trial['metadata']['trial_id']}_Q{i}",
                "question": probe['question'],
                "answer": probe['expected'],
                "type": probe['logic'], # 'Update_Success' or 'Locality_Integrity'
                "domain": trial['metadata']['domain']
            })

        # Final MAB Object
        mab_entry = {
            "trial_id": trial['metadata']['trial_id'],
            "context_chunks": chunks,
            "queries": mab_queries,
            "metadata": {
                "is_negation": trial['metadata']['is_negation'],
                "noise_sequence": s_map['noise_sequence']
            }
        }
        mab_dataset.append(mab_entry)

    return mab_dataset

if __name__ == "__main__":
    input_file = 'integrity_staggered_dataset.json'
    output_file = 'mab_ready_dataset.json'

    if os.path.exists(input_file):
        source = load_dataset(input_file)
        converted = convert_to_mab(source)
        
        with open(output_file, 'w') as f:
            json.dump(converted, f, indent=2)
        print(f"Successfully converted {len(converted)} trials to MAB format.")
    else:
        print(f"Error: {input_file} not found. Generate the dataset first!")