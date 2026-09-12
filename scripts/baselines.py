"""Baselines for the golden evaluation.

Trivial: canned reply, never escalates. Predicts the majority intent class.
Simple:  retrieval-only — copies the top hybrid+rerank historical reply verbatim,
         escalates below a rerank threshold, and predicts intent via the nearest
         Banking77 neighbour's label (kind-filtered search).
"""
from __future__ import annotations

from typing import Any

import config
from src import weaviate_store as ws

TRIVIAL_REPLY = (
    "Hi! Thanks for reaching out to us. Please send us a DM with your account "
    "details so we can take a look and help sort this out for you."
)


def majority_intent(golden_rows: list[dict[str, Any]]) -> str:
    from collections import Counter
    return Counter(r["gold_intent"] for r in golden_rows).most_common(1)[0][0]


def trivial_baseline(message: str, pred_intent: str) -> dict[str, Any]:
    return {
        "bl": "trivial",
        "intent": pred_intent,
        "intent_confidence": 1.0,
        "escalate": False,
        "escalation_reason": "",
        "reply": TRIVIAL_REPLY,
        "grounded_in": "none",
        "from_retrieval": [],
    }


def simple_baseline(message: str, esc_threshold: float = 0.35) -> dict[str, Any]:
    cases = ws.search(message, limit_k=1, kind="twitter")
    best = cases[0] if cases else None
    rerank = best.get("rerank_score") if best else None
    escalate = best is None or rerank is None or rerank < esc_threshold

    intent = "unknown"
    bank = ws.search(message, limit_k=1, kind="banking77")
    if bank and bank[0].get("label"):
        intent = str(bank[0]["label"])

    return {
        "bl": "simple",
        "intent": intent,
        "intent_confidence": rerank if rerank is not None else 0.0,
        "escalate": escalate,
        "escalation_reason": "no similar historical case" if escalate else "",
        "reply": "" if escalate else (best["reply"] or ""),
        "grounded_in": best["customer_tweet_id"] if (best and not escalate) else "none",
        "from_retrieval": [best] if best else [],
        **({"rerank": rerank} if rerank is not None else {}),
    }