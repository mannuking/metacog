"""Tinker API client wrapper for Metacog.

Wraps the real `tinker` Python SDK with a thin facade tailored to our use case:
- Initialize `ServiceClient` (auth, model listing, capability check)
- Lazily create `SamplingClient` per model (for base-model inference)
- (Phase 1) Lazily create `TrainingClient` (LoRA fine-tune, RL)
- Return a `SampleResult` with the decoded text + per-token logprobs

`SampleResult.logprobs` is the REAL confidence proxy: the mean per-token
log-probability of the generated continuation. Higher = more confident.

Environment:
- TINKER_API_KEY loaded from .env at repo root (TINKER_API_KEY=***)

Usage:
    from tink.client import TinkerClient
    client = TinkerClient()
    result = client.sample("Q: What is 2+2?\nA:", model="Qwen/Qwen3-30B-A3B-Instruct-2507", max_tokens=64)
    print(result.text, "logprob_mean=", result.logprob_mean)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root before any tinker imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

import tinker
from tinker import types as ttypes


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MetacogConfig:
    """Defaults for the Metacog project (override per-call)."""

    # Default base model for sampling (Phase 0 baseline). Override with client.sample(model=...)
    # User pick: Qwen 3.6 family, skipping SFT (Option 3 in the planning chat)
    default_base_model: str = os.environ.get(
        "TINKER_BASE_MODEL", "Qwen/Qwen3.6-35B-A3B"
    )
    # Default LoRA config for training (Phase 1+)
    lora_rank: int = 32
    seed: int = 0
    # Default sampling params
    default_max_tokens: int = 1024
    default_temperature: float = 0.7
    default_top_p: float = 0.95
    default_top_k: int = 50


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    """Result of a single Tinker sample call.

    Attributes
    ----------
    text : str
        The decoded completion text.
    prompt : str
        The original prompt text (echoed for traceability).
    model : str
        The model used (base model or LoRA path).
    n_tokens : int
        Number of generated tokens.
    finish_reason : str | None
        Why sampling stopped (StopReason enum value or string).
    latency_s : float
        Wall-clock time for the sample call.
    logprobs : list[float] | None
        Per-token log probabilities (one float per generated token).
    logprob_mean : float
        Mean of `logprobs` over non-None values. The real confidence signal.
    logprob_sum : float
        Sum of `logprobs` over non-None values. The total-information signal.
    raw_response : Any
        The underlying `tinker.SampleResponse` (for advanced consumers).
    """

    text: str
    prompt: str
    model: str
    n_tokens: int = 0
    finish_reason: str | None = None
    latency_s: float = 0.0
    logprobs: list[float] | None = None
    logprob_mean: float = 0.0
    logprob_sum: float = 0.0
    raw_response: Any = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class TinkerClient:
    """Thin wrapper around the Tinker SDK.

    Caches the `ServiceClient` and lazily creates `SamplingClient` per base model.
    """

    def __init__(self, config: MetacogConfig | None = None):
        api_key = os.environ.get("TINKER_API_KEY", "").strip()
        if not api_key or api_key == "PASTE_YOUR_TINKER_API_KEY_HERE":
            raise RuntimeError(
                "TINKER_API_KEY missing. Paste your Tinker console key into "
                f"{_PROJECT_ROOT}/.env (file: TINKER_API_KEY=...)."
            )
        self.config = config or MetacogConfig()
        self._service: tinker.ServiceClient | None = None
        self._sampling_clients: dict[str, tinker.SamplingClient] = {}
        self._tokenizers: dict[str, Any] = {}

    # ---- internal helpers ----------------------------------------------------

    @property
    def service(self) -> tinker.ServiceClient:
        if self._service is None:
            self._service = tinker.ServiceClient()
        return self._service

    def _sampling_for(self, model: str) -> tinker.SamplingClient:
        """Return a cached `SamplingClient` for the given base model."""
        if model not in self._sampling_clients:
            sc = self.service.create_sampling_client(base_model=model)
            # `create_sampling_client` returns an APIFuture; wait for the actual client.
            # In practice, the SDK may return the client directly depending on the mode.
            if hasattr(sc, "result"):
                sc = sc.result()
            self._sampling_clients[model] = sc
            # Tokenizer is per-base-model
            try:
                self._tokenizers[model] = sc.get_tokenizer()
            except Exception:
                self._tokenizers[model] = None
        return self._sampling_clients[model]

    def _tokenizer(self, model: str):
        return self._tokenizers.get(model)

    # ---- public API ----------------------------------------------------------

    def health_check(self) -> dict:
        """Verify the API key works by listing server capabilities.

        Returns a dict: {ok, model, base_model, latency_s, error?}.
        """
        t0 = time.time()
        try:
            caps = self.service.get_server_capabilities()
            return {
                "ok": True,
                "model": str(caps),
                "base_model": self.config.default_base_model,
                "latency_s": round(time.time() - t0, 3),
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "base_model": self.config.default_base_model,
                "latency_s": round(time.time() - t0, 3),
            }

    def sample(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        num_samples: int = 1,
    ) -> list[SampleResult]:
        """Sample one or more completions from a base model.

        Parameters
        ----------
        prompt : str
            The full prompt text (including any chat formatting you want).
        model : str | None
            Base model identifier (defaults to MetacogConfig.default_base_model).
        max_tokens, temperature, top_p, top_k, stop, num_samples :
            Sampling overrides; default to MetacogConfig.

        Returns
        -------
        list[SampleResult]
            One entry per sample (length = num_samples). For Phase 0 baseline,
            pass num_samples=1 for K=1 sampling or num_samples=5 for K=5
            self-consistency.
        """
        model = model or self.config.default_base_model
        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = self.config.default_temperature if temperature is None else temperature
        top_p = self.config.default_top_p if top_p is None else top_p
        top_k = self.config.default_top_k if top_k is None else top_k

        sc = self._sampling_for(model)
        tok = self._tokenizer(model)

        # Build ModelInput. The SDK accepts a list of `ModelInputChunk` (strings, tokens, etc.)
        prompt_tokens = None
        if tok is not None:
            try:
                prompt_tokens = tok.encode(prompt, add_special_tokens=False)
            except Exception:
                prompt_tokens = None

        if prompt_tokens is not None:
            mi = ttypes.ModelInput(chunks=[ttypes.EncodedTextChunk(tokens=prompt_tokens)])
        else:
            mi = ttypes.ModelInput(chunks=[ttypes.EncodedTextChunk(tokens=prompt)])

        sp = ttypes.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )

        t0 = time.time()
        try:
            resp = sc.sample(prompt=mi, num_samples=num_samples, sampling_params=sp)
            if hasattr(resp, "result"):
                resp = resp.result()
        except Exception as e:
            # Surface the upstream error verbatim so the runner can log it.
            raise RuntimeError(f"tinker.sample failed: {type(e).__name__}: {e}") from e
        latency = time.time() - t0

        # Decode each sequence
        results: list[SampleResult] = []
        for seq in resp.sequences:
            token_ids = list(seq.tokens) if seq.tokens is not None else []
            logprobs = list(seq.logprobs) if seq.logprobs is not None else []

            # Decode text
            if tok is not None and token_ids:
                try:
                    text = tok.decode(token_ids, skip_special_tokens=True)
                except Exception:
                    text = "".join(chr(t) if 0 <= t < 0x110000 else "" for t in token_ids)
            else:
                text = ""

            # Confidence signal: mean of per-token logprobs (filter None)
            finite_lp = [lp for lp in logprobs if lp is not None]
            lp_mean = sum(finite_lp) / len(finite_lp) if finite_lp else 0.0
            lp_sum = sum(finite_lp)

            finish_reason = seq.stop_reason.value if hasattr(seq.stop_reason, "value") else str(seq.stop_reason)

            results.append(
                SampleResult(
                    text=text,
                    prompt=prompt,
                    model=model,
                    n_tokens=len(token_ids),
                    finish_reason=finish_reason,
                    latency_s=round(latency / max(num_samples, 1), 4),
                    logprobs=finite_lp,
                    logprob_mean=lp_mean,
                    logprob_sum=lp_sum,
                    raw_response=seq,
                )
            )

        return results

    def list_supported_models(self) -> list[str]:
        """Return the list of supported base models, if the SDK exposes it.

        Falls back to empty list + warning if not available.
        """
        try:
            caps = self.service.get_server_capabilities()
            # Try common attribute names
            for attr in ("supported_models", "models", "base_models"):
                if hasattr(caps, attr):
                    val = getattr(caps, attr)
                    if isinstance(val, (list, tuple)):
                        return [str(m) for m in val]
                    if isinstance(val, dict):
                        return [str(m) for m in val.keys()]
            # Try as dataclass fields
            try:
                import dataclasses
                if dataclasses.is_dataclass(caps):
                    return [f"{f.name}={getattr(caps, f.name)}" for f in dataclasses.fields(caps)]
            except Exception:
                pass
            return [str(caps)]
        except Exception as e:
            return [f"<error: {type(e).__name__}: {e}>"]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """One-shot health check. Raises RuntimeError if TINKER_API_KEY missing."""
    return TinkerClient().health_check()


if __name__ == "__main__":
    import json
    print(json.dumps(health_check(), indent=2))
