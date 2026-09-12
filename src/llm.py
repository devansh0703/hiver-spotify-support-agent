"""OpenAI-compatible NVIDIA NIM chat wrapper (integrate.api.nvidia.com)."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from openai import OpenAI

import config

Client = OpenAI  # type alias kept for compat

_client: OpenAI | None = None

# NIM free-tier rate limits get exhausted by eager concurrency and every 429
# triggers a retry stampede. Hold ALL chat() calls to a global minimum spacing
# (default ~3.0s -> <=20 req/min) so requests never outpace the assigned RPM.
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
MIN_REQUEST_SPACING = float(getattr(config, "MIN_REQUEST_SPACING", 2.0))


def _throttle() -> None:
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        wait = MIN_REQUEST_SPACING - (time.monotonic() - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY,
            timeout=180,
            max_retries=3,
        )
    return _client


def make_nvidia_llm(model: str | None = None, temperature: float | None = None, max_tokens: int = 4096):
    """LlamaIndex LLM for NVIDIA NIM (OpenAI-compatible).

    Plain `OpenAILike` would reject NIM model ids for function calling, so we
    subclass it and advertise the tool-calling capability Nemotron actually has.
    """
    from llama_index.llms.openai_like import OpenAILike

    class NvidiaNIM(OpenAILike):
        @property
        def metadata(self):
            from llama_index.core.llms import LLMMetadata
            return LLMMetadata(
                model_name=self.model,
                is_chat_model=True,
                is_function_calling_model=True,
                context_window=131072,
            )

    return NvidiaNIM(
        api_base=config.NVIDIA_BASE_URL,
        api_key=config.NVIDIA_API_KEY,
        model=model or config.AGENT_MODEL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=max_tokens,
        timeout=300.0,
        is_chat_model=True,
    )


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 1400,
    json_mode: bool = False,
) -> str:
    """Single chat completion, retries on rate-limit / 5xx.

    json_mode=True requests OpenAI-style structured output; Nemotron NIM
    supports it, which cuts the token budget wasted on reasoning blocks.
    """
    model = model or config.AGENT_MODEL
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    extra: dict[str, Any] = {}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
        # Nemotron 3.5 lightning is a reasoning model; thinking competes for the
        # max_tokens budget and truncates the JSON before the closing brace.
        # Disabling thinking guarantees the full budget produces the answer.
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(4):
        try:
            _throttle()
            r = get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            msg = r.choices[0].message
            content = msg.content or msg.reasoning_content or ""
            if content:
                return content
            raise RuntimeError("empty completion")
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            sleep_s = (attempt + 1) * (2 if attempt == 0 else 4)
            # rate limits deserve a long calm-down, not a stampede
            if getattr(exc, "status_code", None) == 429 or "rate limit" in str(exc).lower():
                sleep_s = max(sleep_s, 8 + attempt * 6)
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")


def extract_json(text: str) -> dict[str, Any]:
    """Robustly pull a JSON object out of a model response (handles
    reasoning-model 'thinking' prefixes)."""
    text = text.strip()
    # strip common thinking wrappers
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"Here'?s a thinking process:.*?\n\n", "\n", text, flags=re.S)
    # find every top-level {...} block and keep the LAST (JSON usually at end)
    candidates = re.findall(r"\{.*\}", text, re.S)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no dict JSON in response: {text[:400]}")


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "hi"}], max_tokens=50))