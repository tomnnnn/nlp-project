"""Language models.

``LLM`` is the backend-agnostic interface and ``VLLM`` is the local vLLM
implementation used by the solvers.
"""

from .base import LLM
from .vllm_ import VLLM

__all__ = ["LLM", "VLLM"]
