"""vLLM backend.

Runs an open-weights model from Hugging Face locally on the GPU with vLLM.
The engine is created lazily on first use, so importing the package does not
require a GPU or even the vllm package to be installed.
"""

from typing import Any, Dict, List

from .base import LLM


class VLLM(LLM):
    """Generate text with a local model loaded by vLLM.

    Parameters
    ----------
    model_name:
        Hugging Face model id or path, e.g. ``"Qwen/Qwen2.5-7B-Instruct"``.
    temperature:
        Default sampling temperature. Use ``0`` for greedy decoding.
    max_tokens:
        Default maximum number of generated tokens.
    **engine_kwargs:
        Extra keyword arguments forwarded to ``vllm.LLM`` (for example
        ``tensor_parallel_size``, ``gpu_memory_utilization`` or ``dtype``).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **engine_kwargs: Any,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.max_tokens = max_tokens
        self.engine_kwargs = engine_kwargs
        self._engine = None
        self._tokenizer = None

    @property
    def engine(self):
        """Load the vLLM engine on first access."""
        if self._engine is None:
            from vllm import LLM as VLLMEngine

            self._engine = VLLMEngine(model=self.model_name, **self.engine_kwargs)
            self._tokenizer = self._engine.get_tokenizer()
        return self._engine

    def _sampling_params(self, **kwargs: Any):
        from vllm import SamplingParams

        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        params.update(kwargs)
        return SamplingParams(**params)

    def _render_chat(self, messages: List[Dict[str, str]]) -> str:
        """Apply the model's chat template, with a plain-text fallback."""
        self.engine  # make sure the tokenizer is available
        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.tools:
            template_kwargs["tools"] = self.tools

        if getattr(self._tokenizer, "chat_template", None):
            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Older tokenizers may not accept the ``tools`` argument.
                template_kwargs.pop("tools", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)

        return "\n".join(
            [f"{message['role']}: {message['content']}" for message in messages]
            + ["assistant:"]
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Complete a single prompt and return the generated text."""
        outputs = self.engine.generate([prompt], self._sampling_params(**kwargs))
        return outputs[0].outputs[0].text

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """Complete a conversation and return the assistant message."""
        prompt = self._render_chat(self.with_system_prompt(messages))
        return {"role": "assistant", "content": self.generate(prompt, **kwargs)}
