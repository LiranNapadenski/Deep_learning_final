import argparse
import json
import csv
import os
from tqdm import tqdm
import google.generativeai as genai
from llm_utils import gemini_generate

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
        if "GEMINI_API_KEY" in config:
            os.environ["GEMINI_API_KEY"] = config["GEMINI_API_KEY"]
        if "GITHUB_TOKEN" in config:
            os.environ["GITHUB_TOKEN"] = config["GITHUB_TOKEN"]
        if "ZEP_API_KEY" in config:
            os.environ["ZEP_API_KEY"] = config["ZEP_API_KEY"]
except FileNotFoundError:
    pass

if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    
def llm_judge(question, expected, actual):
    if not actual:
        return False
    prompt = f"""
You are an expert judge evaluating an AI memory system's response.
Question asked: {question}
Expected core information: {expected}
Actual response from system: {actual}

Does the actual response logically contain the expected information and is it factually correct according to the expected information?
Answer with exactly "TRUE" if it does, or "FALSE" if it does not or contradicts it.
"""
    try:
        text = gemini_generate(prompt, model_name="gemini-3.1-flash-lite-preview", temperature=0.0).upper()
        return "TRUE" in text
    except Exception as e:
        print(f"LLM judge evaluation failed: {e}")
        # Fallback to simple string match if API fails
        return expected.lower()[:30] in actual.lower()

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation(system_name, dataset_path, output_csv, trial_limit=None):
    if system_name == "baseline":
        from memory_wrappers.naive_baseline import NaiveBaselineWrapper
        memory_module = NaiveBaselineWrapper()
    elif system_name == "mem0":
        from memory_wrappers.mem0_wrapper import Mem0Wrapper
        memory_module = Mem0Wrapper()
    elif system_name == "memgpt":
        from memory_wrappers.memgpt_wrapper import MemGPTWrapper
        memory_module = MemGPTWrapper()
    elif system_name == "zep":
        from memory_wrappers.zep_wrapper import ZepWrapper
        memory_module = ZepWrapper()
    elif system_name == "rag":
        from memory_wrappers.rag_wrapper import RAGMemoryWrapper
        memory_module = RAGMemoryWrapper()
    else:
        raise ValueError(f"System {system_name} is not supported.")
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    
    if trial_limit:
        dataset = dataset[:trial_limit]
        print(f"Limiting evaluation to {trial_limit} trials.")
    
    print(f"Exporting results incrementally to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "system", "trial_id", "domain", "is_negation", 
            "probe_logic", "question", "expected", "actual", "is_correct"
        ])
        writer.writeheader()
        
        for i, trial in enumerate(tqdm(dataset, desc=f"Evaluating {system_name}")):
            trial_id = trial['metadata']['trial_id']
            domain = trial['metadata']['domain']
            is_negation = trial['metadata']['is_negation']
            
            # Reset memory for the new trial
            memory_module.reset()
            if hasattr(memory_module, 'add_history'):
                memory_module.add_history(trial['session_history'])
            else:
                # Feed session history
                for turn in trial['session_history']:
                    memory_module.add_turn(turn['role'], turn['content'])
            
            # Evaluate probes
            for probe in trial['evaluation_probes']:
                question = probe['question']
                expected_answer = probe['expected']
                logic = probe['logic']  # 'Update_Success' or 'Locality_Integrity'
                
                # Query the memory system
                actual_answer = memory_module.query(question)
                
                # Use LLM judge for grading
                is_correct = llm_judge(question, expected_answer, actual_answer)
                
                writer.writerow({
                    "system": system_name,
                    "trial_id": trial_id,
                    "domain": domain,
                    "is_negation": is_negation,
                    "probe_logic": logic,
                    "question": question,
                    "expected": expected_answer,
                    "actual": actual_answer,
                    "is_correct": is_correct
                })
                f.flush()
                
    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate memory systems on Conflict Interference Dataset.")
    parser.add_argument("--system", type=str, required=True, choices=["baseline", "mem0", "memgpt", "zep", "rag"],
                        help="The memory system to evaluate.")
    parser.add_argument("--dataset", type=str, default="conflict_interference_dataset.json",
                        help="Path to the JSON dataset.")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Path to the output CSV file.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of trials for a quick dry run test.")
    
    args = parser.parse_args()
    
    # Ensure GEMINI_API_KEY is available if we are using baseline
    if args.system == "baseline" and not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY environment variable is not set. The naive baseline wrapper requires it.")
        
    run_evaluation(args.system, args.dataset, args.output, args.limit)
