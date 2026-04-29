import uuid
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from .base_wrapper import BaseMemoryWrapper
from llm_utils import gemini_generate


class RAGMemoryWrapper(BaseMemoryWrapper):
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384

        # FAISS index (cosine similarity via inner product)
        self.index = faiss.IndexFlatIP(self.dim)

        self.texts = []
        self.metadata = []
        self.user_id = None

    def reset(self):
        """Fresh memory per trial"""
        self.user_id = f"trial_{uuid.uuid4().hex[:8]}"
        self.index.reset()
        self.texts = []
        self.metadata = []

    # ---------- Internal ----------
    def _embed(self, texts: List[str]):
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        return np.array(embeddings).astype("float32")

    def _add_texts(self, texts: List[str]):
        embeddings = self._embed(texts)
        self.index.add(embeddings)
        self.texts.extend(texts)

    # ---------- Public API ----------
    def add_history(self, history: list):
        """
        Store conversation as chunks.
        Each turn becomes a retrievable unit.
        """
        chunks = []
        for m in history:
            role = m["role"]
            content = m["content"]
            chunks.append(f"{role}: {content}")

        self._add_texts(chunks)

    def add_turn(self, role: str, content: str):
        """Add incremental update"""
        chunk = f"{role}: {content}"
        self._add_texts([chunk])

    def query(self, question: str) -> str:
        try:
            # 1. Retrieve relevant chunks
            q_emb = self._embed([question])
            scores, indices = self.index.search(q_emb, k=min(20, len(self.texts)))

            retrieved = []
            for idx in indices[0]:
                if idx < len(self.texts):
                    retrieved.append(self.texts[idx])

            context = "\n".join([f"- {t}" for t in retrieved])

            # 2. Prompt (handles corrections explicitly)
            prompt = (
                "You are an expert judge.\n"
                "Use the retrieved conversation snippets below.\n"
                "Answer precisely.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

            return gemini_generate(
                prompt,
                model_name="gemini-3.1-flash-lite-preview",
                temperature=0.0
            )

        except Exception as e:
            return f"RAG Error: {e}"