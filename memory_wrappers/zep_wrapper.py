import os
import uuid
import time
from zep_cloud.client import Zep
from zep_cloud import Message
from .base_wrapper import BaseMemoryWrapper
from llm_utils import gemini_generate

class ZepWrapper(BaseMemoryWrapper):
    def __init__(self):
        self.api_key = os.environ.get("ZEP_API_KEY")
        self.client = Zep(api_key=self.api_key)
        self.user_id = None
        self.session_id = None
        self._pending_messages = []

    def reset(self):
        """Creates unique User/Thread for 100% trial isolation."""
        self.user_id = f"u_{uuid.uuid4().hex[:8]}"
        self.session_id = f"t_{uuid.uuid4().hex[:8]}"
        self._pending_messages = []
        
        try:
            self.client.user.add(user_id=self.user_id)
            self.client.thread.create(user_id=self.user_id, thread_id=self.session_id)
        except Exception as e:
            print(f"  [Zep] Reset error: {e}")

    def add_turn(self, role: str, content: str):
        self._pending_messages.append(Message(role=role, content=content))

    def add_history(self, history: list):
        """Batch upload history in chunks to satisfy Zep Cloud limits."""
        messages = [Message(role=m['role'], content=m['content']) for m in history]
        try:
            # Chunking logic for 30-message limit
            chunk_size = 25
            for i in range(0, len(messages), chunk_size):
                chunk = messages[i:i + chunk_size]
                self.client.thread.add_messages(self.session_id, messages=chunk)
            
            # Allow the Graphiti engine to process the 33-turn block
            self._wait_for_facts(timeout=45) 
        except Exception as e:
            print(f"  [Zep] History upload error: {e}")

    def _wait_for_facts(self, timeout=45):
        start = time.time()
        while time.time() - start < timeout:
            try:
                res = self.client.graph.search(
                    user_id=self.user_id,
                    query="user",
                    limit=5
                )

                if res and (res.edges or res.context):
                    return True

            except Exception:
                pass

            time.sleep(3)

        print("[Zep] Warning: Graph may not be fully ready")
        return False
    def query(self, question: str) -> str:
        try:
            # 1. Sync any updates
            if self._pending_messages:
                self.client.thread.add_messages(self.session_id, messages=self._pending_messages)
                self._pending_messages = []
                self._wait_for_facts(timeout=25)

            # 2. THE FIX: Broaden the search scope
            # We use 'rerank=True' if available in your SDK version, 
            # and we ask for a wider center of gravity.
            expanded_query = f"""
                {question}
                Relevant user facts, personal info, corrections, dates, preferences.
                """

            search_results = self.client.graph.search(
                user_id=self.user_id,
                query=expanded_query,
                center_node_id=None,
                limit=20
            )
            
            # 3. Aggressive Context Building
            # If search_results.context is empty, it's a failure of the Graphiti extractor.
            # We will manually combine everything Zep found.
            context_pieces = []
            if search_results.context:
                context_pieces.append(f"Narrative Summary: {search_results.context}")
            
            if search_results.edges:
                # We sort edges by 'created_at' if available, or just list them
                # Latest facts should ideally be at the bottom for the LLM's recency bias
                edge_list = [f"- {e.fact}" for e in search_results.edges if e.fact]
                context_pieces.append("Atomic Facts:\n" + "\n".join(edge_list))

            if not context_pieces:
                try:
                    thread = self.client.thread.get(self.session_id)
                    msgs = thread.messages[-20:] if thread.messages else []

                    fallback = "\n".join([f"{m.role}: {m.content}" for m in msgs])
                    context_pieces.append("Raw Conversation:\n" + fallback)

                except Exception:
                    pass
            
            world_state = "\n\n".join(context_pieces)

            # 4. STAGE-GATE PROMPT
            # We add a 'Strictness' clause to stop hallucinations
            prompt = (
                "SYSTEM INSTRUCTION: Use the provided Knowledge Graph State to answer.\n"
                "If the state is empty or does not mention the specific detail (like a chair or a date), "
                "say 'Information not found in memory.'\n"
                f"KNOWLEDGE GRAPH STATE:\n{world_state}\n\n"
                f"USER QUESTION: {question}\n"
                "ANSWER:"
            )
            
            return gemini_generate(prompt, model_name="gemini-3.1-flash-lite-preview", temperature=0.0)

        except Exception as e:
            return f"Query Error: {e}"