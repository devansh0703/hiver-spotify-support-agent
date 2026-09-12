"""End-to-end smoke of the LlamaIndex-orchestrated Spotify support agent.

Run:  PYTHONPATH=/home/devansh/hiver /usr/bin/python3 scripts/agent_test.py
"""
from __future__ import annotations

import config
from src.orchestrator import SupportAgentRunner


def main() -> None:
    agent = SupportAgentRunner()
    cases = [
        "my premium subscription got charged twice now i want a refund",
        "the app keeps crashing every time i open a playlist, fix this now",
        "I just got locked out of my account after changing my password",
    ]
    for msg in cases:
        resp = agent.respond(msg)
        reply = resp.get("text", "")
        print("CASE:", msg)
        print("REPLY:", reply[:400].replace("\n", " | "))
        print("-" * 60)


if __name__ == "__main__":
    main()