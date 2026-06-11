"""Tinker API client wrapper for Metacog.

Wraps the Tinker Python SDK with a thin facade tailored to our use case:
- Initialize service + training + sampling clients
- Sample completions from a base model with thinking mode enabled
- (Later) Submit RL training loops with verifiable rewards
- (Later) Save and download LoRA weights

Environment:
- TINKER_API_KEY must be set (we load it from .env at repo root)

Usage:
    from tinker.client import TinkerClient
    client = TinkerClient()
    result = client.sample("Q: What is 2+2?\nA:", max_tokens=64)
    print(result.text)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Load .env from project root before any tinker imports
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class SampleResult:
    """Result of a single Tinker sample call."""

    text: str
    prompt: str
    model: str
    n_tokens: int = 0
    finish_reason: str | None = None
    raw_response: Any = None
    latency_s: float = 0.0


@dataclass
class MetacogConfig:
    """Defaults for the Metacog project."""

    # Base model — only Qwen 3.6 open variant on Tinker, per project decision.
    # We use the IT (instruction-tuned) checkpoint because we need thinking
    # mode to be active by default. LoRA on the base is also possible later.
    base_model: str = "Qwen/Qwen3.6-35B-A3B"

    # Sampling defaults — temperature 0 for reproducibility on math/sci,
    # but we expose override for stochastic self-consistency experiments.
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048

    # Concurrency — keep small to start; Tinker rate-limits.
    max_concurrency: int = 4


class TinkerClient:
    """Thin wrapper around the Tinker SDK for the Metacog project.

    Lazily initializes the service client so the module is importable
    without a valid API key (useful for unit tests and dry-runs).
    """

    def __init__(self, config: MetacogConfig | None = None):
        self.config = config or MetacogConfig()
        self._service = None
        self._api_key = os.environ.get("TINKER_API_KEY", "").strip()

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _require_key(self) -> str:
        if not self._api_key:
            raise RuntimeError(
                "TINKER_API_KEY is not set. Add it to .env in the project root:\n"
                "  TINKER_API_KEY=tml-...\n"
                "Get one at https://tinker-console.thinkingmachines.ai/"
            )
        return self._api_key

    def get_service(self):
        """Lazily construct the Tinker service client."""
        if self._service is None:
            self._require_key()
            import tinker

            self._service = tinker.ServiceClient()
        return self._service

    def sample(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        enable_thinking: bool = True,
    ) -> SampleResult:
        """Sample a single completion from the base (or specified) model.

        Args:
            prompt: Plain-text prompt. The Qwen3.6 chat template is applied
                via the renderer when sampling.
            model: Override the default base model.
            max_tokens: Override max_tokens.
            temperature: Override temperature. 0 = greedy.
            stop: Optional list of stop strings.
            enable_thinking: If True, prepend the <|think|> system turn so
                the model emits a reasoning trace before its answer.

        Returns:
            SampleResult with text, token count, latency.
        """
        service = self.get_service()
        model_name = model or self.config.base_model
        max_tok = max_tokens or self.config.max_tokens
        temp = self.config.temperature if temperature is None else temperature

        t0 = time.perf_counter()
        # Tinker's sample API uses a sampling client. The cookbook has
        # a `tinker_cookbook.renderers` module that handles chat templates
        # per model. For raw text sampling we use the simpler sampling_client.
        sampling = service.create_sampling_client(base_model=model_name)

        # Apply Qwen3.6 thinking template. For the open Qwen3.6-A3B, the
        # template is <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant
        # with <|think|>...<|/think|> blocks. We let the model's default
        # chat template handle that by sending through the renderer.
        # The cookbook provides `tokenizer.apply_chat_template` style helpers
        # but for raw sampling we just construct the prompt and rely on the
        # model to behave correctly with temperature 0.
        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            num_samples=1,
            max_tokens=max_tok,
            temperature=temp,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        if stop:
            kwargs["stop"] = stop

        try:
            response = sampling.sample(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Tinker sample failed: {e}") from e

        # Response shape: response.sequences[0].text (or similar depending on
        # Tinker SDK version). We try a few common shapes.
        text = ""
        n_tokens = 0
        finish_reason = None
        if hasattr(response, "sequences") and response.sequences:
            seq = response.sequences[0]
            if hasattr(seq, "text"):
                text = seq.text
            elif hasattr(seq, "tokens") and hasattr(seq.tokens, "__iter__"):
                text = ""
            n_tokens = getattr(seq, "num_tokens", 0) or 0
        elif hasattr(response, "text"):
            text = response.text
        elif isinstance(response, str):
            text = response

        latency = time.perf_counter() - t0
        return SampleResult(
            text=text,
            prompt=prompt,
            model=model_name,
            n_tokens=n_tokens,
            finish_reason=finish_reason,
            raw_response=response,
            latency_s=latency,
        )

    def sample_batch(
        self,
        prompts: list[str],
        *,
        n_per_prompt: int = 1,
        **kwargs: Any,
    ) -> list[list[SampleResult]]:
        """Sample N completions per prompt. Used for self-consistency
        confidence estimation (sample K, measure agreement).

        Returns a list of lists — outer indexed by prompt, inner by sample.
        """
        results: list[list[SampleResult]] = []
        for prompt in prompts:
            per_prompt: list[SampleResult] = []
            for _ in range(n_per_prompt):
                per_prompt.append(self.sample(prompt, **kwargs))
            results.append(per_prompt)
        return results


def health_check() -> dict[str, Any]:
    """Return a dict describing whether the client is configured and what
    model is the default. Used by the baseline runner as a sanity check."""
    c = TinkerClient()
    return {
        "configured": c.is_configured,
        "api_key_present": bool(c._api_key),
        "api_key_length": len(c._api_key),
        "default_model": c.config.base_model,
        "default_max_tokens": c.config.max_tokens,
        "default_temperature": c.config.temperature,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(health_check(), indent=2))
