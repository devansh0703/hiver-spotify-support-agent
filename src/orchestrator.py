"""LlamaIndex agent orchestration over NVIDIA hardware (Nemotron 3.5 lightning).

The support agent is a LlamaIndex `AgentRunner` (function-calling agent
worker) with a single Weaviate-backed tool that runs the real pipeline in one
deterministic pass — classify -> hybrid+MMR+rerank retrieval -> grounded draft
-> route decision. Single-tool flow avoids slow multi-round-trip tool loops on
the shared NIM endpoint while still being genuine LlamaIndex orchestration.

LLM: nvidia/nemotron-3.5-lightning-30b-a3b via the OpenAI-compatible NIM
adapter (src.llm.make_nvidia_llm). The same pipeline is reachable non-agent-
style via src.agent.SpotifyAgent for eval harnesses that need deterministic
JSON.
"""
from __future__ import annotations

import json

from llama_index.core import Settings
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool

import config
from src.agent import SpotifyAgent
from src.llm import make_nvidia_llm

HANDLE_DESCRIPTION = (
    "Handle a full customer support case: classify the message, retrieve how "
    "{brand} historically resolved similar issues, draft a grounded reply, and "
    "decide auto-handle vs escalate. Pass the latest customer message text."
)


def _handle_case(message: str, prior_turns: str = "") -> str:
    prior = [t for t in prior_turns.split("\n") if t.strip()] or None
    out = SpotifyAgent().respond(message, prior_turns=prior)
    return json.dumps(
        {
            "intent": out["intent"],
            "confidence": round(out["intent_confidence"], 3),
            "escalate": out["escalate"],
            "escalation_reason": out["escalation_reason"],
            "reply": out["reply"],
            "grounded_in": out["grounded_in"],
        },
        ensure_ascii=False,
    )


SYSTEM_PROMPT = (
    "You are {brand} social care on Twitter. For an incoming message, call "
    "handle_support_case(message) once, then relay the final reply to the user. "
    "If escalate is true, tell the user it is being routed to a human specialist "
    "and give the escalation reason. Keep any intro to at most one short line."
)


class SupportAgentRunner:
    """LlamaIndex AgentRunner over a single Weaviate-backed support tool."""

    def __init__(self, model: str | None = None) -> None:
        llm = make_nvidia_llm(
            model=model or config.AGENT_MODEL, max_tokens=4096
        )
        Settings.llm = llm
        worker = FunctionCallingAgentWorker.from_tools(
            [FunctionTool.from_defaults(
                fn=_handle_case,
                name="handle_support_case",
                description=HANDLE_DESCRIPTION.format(brand=config.BRAND_NAME),
            )],
            llm=llm,
            system_prompt=SYSTEM_PROMPT.format(brand=config.BRAND_NAME),
            max_function_calls=2,
        )
        self.runner = AgentRunner(worker)

    def respond(self, message: str, prior_turns: list[str] | None = None) -> dict:
        ptxt = "\n".join(f"- {t[:180]}" for t in (prior_turns or [])[-6:])
        out = self.runner.chat(
            f"Customer message: {message}\nPrior turns: {ptxt or '(none)'}"
        )
        return {"text": str(out).strip(), "message": message}


if __name__ == "__main__":
    agent = SupportAgentRunner()
    r = agent.respond("my premium got charged twice, I want a refund", prior_turns=[])
    print(r["text"][:800])