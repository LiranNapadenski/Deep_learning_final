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
        if "GEMINI_RPM" in config:
            os.environ["GEMINI_RPM"] = str(config["GEMINI_RPM"])
        if "GEMINI_MIN_INTERVAL_SECONDS" in config:
            os.environ["GEMINI_MIN_INTERVAL_SECONDS"] = str(config["GEMINI_MIN_INTERVAL_SECONDS"])
except FileNotFoundError:
    pass

if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    
def llm_judge(question, expected, actual):
    if not actual:
        return False
    prompt = f"""You are an expert evaluator tasked with assessing how well an LLM output addresses its input based on a ground truth.

## Evaluation Criteria:

1. Analyze the input question and ground truth to understand what is being asked and what facts must be present.
2. Examine the output to see what information is provided.
3. Determine if the output directly addresses the question and factually matches the ground truth.
4. Check for contradictory or entirely missing core information.
5. Minor irrelevant details or natural language variations are perfectly acceptable as long as the core ground truth is met.

## Input Question:
{question}

## Ground Truth:
{expected}

## Output to Evaluate:
{actual}

## Examples:

Ground Truth: "No money sent; wire cancelled."
Output: "The landlord waived the bill, and you cancelled the wire transfer entirely."
Verdict: <VERDICT>TRUE</VERDICT> (Correct, the core outcome matches).

Ground Truth: "Car is fixed and with the user."
Output: "The mechanic found a used part."
Verdict: <VERDICT>FALSE</VERDICT> (Incorrect, the car is not yet fixed/returned in this response).

## Evaluation Instructions:

Compare the Output to Evaluate against the Ground Truth.
Does the Output factually contain the core information of the Ground Truth without contradicting it?

First, provide a brief reasoning for your decision.
Then, on a new line, provide your final answer wrapped in <VERDICT> tags, like <VERDICT>TRUE</VERDICT> or <VERDICT>FALSE</VERDICT>.
"""
    try:
        text = gemini_generate(prompt, model_name="gemini-3.1-flash-lite-preview", temperature=0.0).upper()
        return "<VERDICT>TRUE</VERDICT>" in text
    except Exception as e:
        print(f"LLM judge evaluation failed: {e}")
        # Fallback to simple string match if API fails
        return expected.lower()[:30] in actual.lower()

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation(system_name, dataset_path, output_csv, trial_limit=None):
    if system_name == "long_context":
        from memory_wrappers.long_context_wrapper import LongContextWrapper
        memory_module = LongContextWrapper()
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
    elif system_name == "a-mem":
        from memory_wrappers.a_mem_wrapper import AMemWrapper
        memory_module = AMemWrapper()
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
            "system", "trial_id", "domain", "is_over_updating", 
            "probe_logic", "question", "expected", "actual", "is_correct"
        ])
        writer.writeheader()
        
        for i, trial in enumerate(tqdm(dataset, desc=f"Evaluating {system_name}")):
            trial_id = trial['metadata']['trial_id']
            domain = trial['metadata']['domain']
            is_over_updating = trial['metadata'].get('is_over_updating', trial.get('is_over_updating', 'None'))
            
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
                
                # Clean actual_answer to ensure one trial per line in CSV
                if actual_answer:
                    actual_answer = actual_answer.replace("\n", " ").replace("\r", " ").strip()
                
                # Use LLM judge for grading
                is_correct = llm_judge(question, expected_answer, actual_answer)
                
                writer.writerow({
                    "system": system_name,
                    "trial_id": trial_id,
                    "domain": domain,
                    "is_over_updating": is_over_updating,
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
    parser.add_argument("--system", type=str, required=True, choices=["long_context", "mem0", "memgpt", "zep", "rag", "a-mem"],
                        help="The memory system to evaluate.")
    parser.add_argument("--dataset", type=str, default="conflict_interference_dataset.json",
                        help="Path to the JSON dataset.")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Path to the output CSV file.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of trials for a quick dry run test.")
    
    args = parser.parse_args()
    
    # Ensure GEMINI_API_KEY is available if we are using long_context
    if args.system == "long_context" and not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY environment variable is not set. The long context wrapper requires it.")
        
    run_evaluation(args.system, args.dataset, args.output, args.limit)
