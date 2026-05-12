from .base_wrapper import BaseMemoryWrapper
import os
from llm_utils import gemini_generate

class LongContextWrapper(BaseMemoryWrapper):
    """
    A long context wrapper that just concatenates all history and passes it to GitHub Models.
    """
    def __init__(self):
        self.history = []
        
    def reset(self):
        self.history = []

    def add_turn(self, role: str, content: str):
        # We store the conversation linearly
        self.history.append(f"{role.capitalize()}: {content}")

    def query(self, question: str) -> str:
        # Pass the full history as requested, representing the upper bound baseline.
        context = "\n".join(self.history)
        prompt = f"Given the following recent conversation history:\n\n{context}\n\nAnswer the following question clearly and concisely based ONLY on the provided recent history.\nQuestion: {question}\nAnswer:"
        
        try:
            return gemini_generate(prompt, model_name="gemini-3.1-flash-lite-preview", temperature=0.0)
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return "Error generating response"
