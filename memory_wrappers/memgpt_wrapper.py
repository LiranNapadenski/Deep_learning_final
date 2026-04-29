import uuid
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from .base_wrapper import BaseMemoryWrapper
from llm_utils import gemini_generate


class MemGPTWrapper(BaseMemoryWrapper):
    def __init__(self):
        self.user_id = None

        # Working memory (short-term)
        self.working_memory = []

        # Long-term memory (vector store)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384
        self.index = faiss.IndexFlatIP(self.dim)

        self.long_term_texts = []

    # -------------------------
    # Lifecycle
    # -------------------------
    def reset(self):
        self.user_id = f"trial_{uuid.uuid4().hex[:8]}"
        self.working_memory = []
        self.long_term_texts = []
        self.index.reset()

    # -------------------------
    # Embedding
    # -------------------------
    def _embed(self, texts: List[str]):
        emb = self.embedder.encode(texts, normalize_embeddings=True)
        return np.array(emb).astype("float32")

    def _store_long_term(self, text: str):
        emb = self._embed([text])
        self.index.add(emb)
        self.long_term_texts.append(text)

    # -------------------------
    # Memory Controller
    # -------------------------
    def _should_store(self, content: str) -> bool:
        """
        Decide if something is worth long-term storage.
        You can make this smarter later.
        """
        keywords = ["birthday", "name", "age", "live", "prefer", "favorite"]
        return any(k in content.lower() for k in keywords)

    def _format_turn(self, role: str, content: str):
        return f"{role}: {content}"

    # -------------------------
    # Public API
    # -------------------------
    def add_history(self, history: list):
        for m in history:
            self.add_turn(m["role"], m["content"])

    def add_turn(self, role: str, content: str):
        formatted = self._format_turn(role, content)

        # 1. Always add to working memory
        self.working_memory.append(formatted)

        # Keep last N turns (simulate context window)
        if len(self.working_memory) > 10:
            self.working_memory.pop(0)

        # 2. Decide if it goes to long-term memory
        if self._should_store(content):
            # Store as explicit fact (important!)
            fact = f"Fact: {content}"
            self._store_long_term(fact)

    # -------------------------
    # Retrieval
    # -------------------------
    def _retrieve_long_term(self, question: str, k=10):
        if len(self.long_term_texts) == 0:
            return []

        q_emb = self._embed([question])
        scores, indices = self.index.search(q_emb, k=min(k, len(self.long_term_texts)))

        results = []
        for idx in indices[0]:
            if idx < len(self.long_term_texts):
                results.append(self.long_term_texts[idx])

        return results

    # -------------------------
    # Query
    # -------------------------
    def query(self, question: str) -> str:
        try:
            # 1. Retrieve long-term memory
            long_term = self._retrieve_long_term(
                f"{question} (personal facts, corrections, preferences)"
            )

            # 2. Build context
            working_context = "\n".join(self.working_memory)
            long_term_context = "\n".join([f"- {m}" for m in long_term])

            # 3. Prompt (MemGPT-style reasoning)
            prompt = (
                "You are an intelligent assistant with memory.\n\n"

                "Short-term memory (recent conversation):\n"
                f"{working_context}\n\n"

                "Long-term memory (stored facts):\n"
                f"{long_term_context}\n\n"

                "Instructions:\n"
                "- Use long-term memory for facts.\n"
                "- If there are corrections, prefer the most recent.\n"
                "- Do not say 'unknown' if the answer can be inferred.\n\n"

                f"Question: {question}\n"
                "Answer:"
            )

            return gemini_generate(
                prompt,
                model_name="gemini-3.1-flash-lite-preview",
                temperature=0.0
            )

        except Exception as e:
            return f"MemGPT Error: {e}"