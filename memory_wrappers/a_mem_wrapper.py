import uuid
import time
from typing import List, Dict
from .base_wrapper import BaseMemoryWrapper
from llm_utils import gemini_generate


class AMemWrapper(BaseMemoryWrapper):
    def __init__(self):
        self.user_id = None
        self.memory = []  # list of atomic facts

    def reset(self):
        self.user_id = f"trial_{uuid.uuid4().hex[:8]}"
        self.memory = []

    # -------------------------
    # Fact Extraction
    # -------------------------
    def _extract_facts(self, text: str) -> List[Dict]:
        """
        Use LLM to convert text into atomic facts.
        """
        prompt = (
            "Extract atomic facts from the text.\n"
            "Return JSON list of objects with:\n"
            "subject, relation, value\n\n"
            f"Text: {text}\n\n"
            "Example:\n"
            '[{"subject":"user","relation":"birthday","value":"Sunday"}]\n\n'
            "Output JSON:"
        )

        response = gemini_generate(
            prompt,
            model_name="gemini-3.1-flash-lite-preview",
            temperature=0.0
        )

        try:
            import json
            facts = json.loads(response)
            return facts if isinstance(facts, list) else []
        except:
            return []

    # -------------------------
    # Storage
    # -------------------------
    def _store_fact(self, fact: Dict):
        fact["timestamp"] = time.time()

        # Remove conflicting facts (same subject + relation)
        self.memory = [
            f for f in self.memory
            if not (f["subject"] == fact["subject"] and f["relation"] == fact["relation"])
        ]

        self.memory.append(fact)

    # -------------------------
    # Public API
    # -------------------------
    def add_history(self, history: list):
        for m in history:
            self.add_turn(m["role"], m["content"])

    def add_turn(self, role: str, content: str):
        facts = self._extract_facts(content)
        for fact in facts:
            self._store_fact(fact)

    # -------------------------
    # Retrieval
    # -------------------------
    def _retrieve_relevant(self, question: str):
        """
        Simple semantic filtering using LLM.
        """
        if not self.memory:
            return []

        facts_text = "\n".join([
            f"{f['subject']} {f['relation']} {f['value']}"
            for f in self.memory
        ])

        prompt = (
            "Select relevant facts for answering the question.\n\n"
            f"Facts:\n{facts_text}\n\n"
            f"Question: {question}\n\n"
            "Return the relevant facts as text:"
        )

        response = gemini_generate(
            prompt,
            model_name="gemini-3.1-flash-lite-preview",
            temperature=0.0
        )

        return response

    # -------------------------
    # Query
    # -------------------------
    def query(self, question: str) -> str:
        try:
            relevant_facts = self._retrieve_relevant(question)

            prompt = (
                "You are an expert judge.\n"
                "Use the facts below.\n"
                "Facts are already conflict-resolved (latest is correct).\n\n"
                f"{relevant_facts}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

            return gemini_generate(
                prompt,
                model_name="gemini-3.1-flash-lite-preview",
                temperature=0.0
            )

        except Exception as e:
            return f"A-Mem Error: {e}"