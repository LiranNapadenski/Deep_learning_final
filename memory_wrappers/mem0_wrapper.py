import uuid
import os
import time
import random
from mem0 import Memory
from .base_wrapper import BaseMemoryWrapper
from llm_utils import gemini_generate


class Mem0Wrapper(BaseMemoryWrapper):
    def __init__(self):
        # Ensure Gemini API key is set
        if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": "./qdrant_db",
                    "embedding_model_dims": 384,
                }
            },
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-3.1-flash-lite-preview",
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                }
            }
        }

        self.memory = Memory.from_config(config)
        self.user_id = None

    # -------------------------
    # Lifecycle
    # -------------------------
    def reset(self):
        """Fresh identity per trial"""
        self.user_id = f"trial_{uuid.uuid4().hex[:8]}"

    # -------------------------
    # Utilities
    # -------------------------
    def _add_with_retry(self, data, max_retries=5):
        for i in range(max_retries):
            try:
                self.memory.add(data, user_id=self.user_id)
                return
            except Exception as e:
                msg = str(e).lower()
                if any(x in msg for x in ["503", "unavailable", "429", "quota"]):
                    sleep_time = min(10, (2 ** i) + random.random())
                    print(f"[Mem0 retry {i+1}] sleeping {sleep_time:.2f}s → {e}")
                    time.sleep(sleep_time)
                else:
                    raise
        raise RuntimeError("Mem0 add failed after retries")

    def _format_messages(self, history):
        """Normalize messages + hint important facts"""
        messages = []
        for m in history:
            role = m["role"]
            content = m["content"]

            # Hint to improve fact extraction
            if any(k in content.lower() for k in ["birthday", "name", "age", "gym", "session", "time"]):
                content = f"[Potential Fact] {content}"

            messages.append({
                "role": role,
                "content": content
            })
        return messages

    def _ensure_list(self, x):
        """Normalize Mem0 outputs into list form"""
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, dict):
            if "results" in x:
                return x["results"]
            if "memories" in x:
                return x["memories"]
            return [x]
        return [x]

    # -------------------------
    # Public API
    # -------------------------
    def add_history(self, history: list):
        messages = self._format_messages(history)
        self._add_with_retry(messages)

    def add_turn(self, role: str, content: str):
        message = self._format_messages([{"role": role, "content": content}])
        self._add_with_retry(message)

    def query(self, question: str) -> str:
        try:
            # -------------------------
            # 1. Semantic search
            # -------------------------
            expanded_query = f"""
            {question}
            Relevant personal facts, schedules, times, preferences, corrections.
            """

            memories = self.memory.search(
                query=expanded_query,
                filters={"user_id": self.user_id},
                limit=20
            )

            memories = self._ensure_list(memories)

            # -------------------------
            # 2. Fallback to all memories
            # -------------------------
            if len(memories) < 3:
                all_memories = self.memory.get_all(filters={"user_id": self.user_id})
                all_memories = self._ensure_list(all_memories)
                memories = memories + all_memories

            # -------------------------
            # 3. Extract text safely
            # -------------------------
            facts = []
            for m in memories:
                if isinstance(m, dict):
                    facts.append(
                        m.get("text") or
                        m.get("memory") or
                        m.get("content") or
                        str(m)
                    )
                else:
                    facts.append(str(m))

            # Deduplicate
            facts = list(dict.fromkeys(facts))

            context = "\n".join(f"- {f}" for f in facts[:30])

            # -------------------------
            # 4. Prompt
            # -------------------------
            prompt = (
                "You are an expert judge.\n"
                "Use ONLY the facts below.\n"
                "If facts conflict, prefer the most recent correction.\n"
                "Do NOT say 'unknown' if the answer can be inferred.\n\n"
                f"Facts:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

            return gemini_generate(
                prompt,
                model_name="gemini-3.1-flash-lite-preview",
                temperature=0.0
            )

        except Exception as e:
            return f"Mem0 Error: {e}"