from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseMemoryWrapper(ABC):
    """
    Abstract base class for all memory system wrappers.
    """
    
    @abstractmethod
    def reset(self):
        """
        Reset the memory module to an empty state for a new trial.
        """
        pass

    @abstractmethod
    def add_turn(self, role: str, content: str):
        """
        Add a conversational turn to the memory system.
        :param role: 'user' or 'assistant'
        :param content: The text content of the turn
        """
        pass

    @abstractmethod
    def query(self, question: str) -> str:
        """
        Query the memory system with a question and return the agent's answer.
        :param question: The question to ask the agent
        :return: The textual answer
        """
        pass
