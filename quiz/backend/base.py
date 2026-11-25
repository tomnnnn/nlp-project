"""The language-model interface.

Every backend implements the same small interface so solvers never depend on
which engine is running underneath. All model code lives in this package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLM(ABC):
    """Common interface for text generation backends."""

    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = "You are a helpful assistant."
        self.tools: List[Any] = []

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def set_tools(self, tools: List[Any]) -> None:
        self.tools = tools

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Stateless single prompt completion."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
        """Stateful conversation completion."""

    def with_system_prompt(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepend the system prompt unless the messages already contain one."""
        if any(message.get("role") == "system" for message in messages):
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages

    def prepare_messages(
        self, user_input: str, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Helper to inject the system prompt and append the new user message."""
        return self.with_system_prompt(
            history + [{"role": "user", "content": user_input}]
        )
