"""Automated metrics over golden-set predictions (pure functions, no IO)."""
from __future__ import annotations

from typing import Any


def _bin(yes: bool) -> int:
    return 1 if yes else 0


def binary_metrics(gold: list[int], pred: list[int]) -> dict[str, Any]:
    assert len(gold) == len(pred)
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    tn = sum(1 for g, p in zip(gold, pred) if not g and not p)
    acc = (tp + tn) / len(gold) if gold else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rate = sum(pred) / len(pred) if pred else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "escalate_rate": round(rate, 4),
    }


def intent_accuracy(pred: list[str], gold: list[str]) -> dict[str, Any]:
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    return {"accuracy": round(correct / len(gold), 4) if gold else 0.0, "n": len(gold)}


def reply_presence(golds: list[int], replies: list[str]) -> dict[str, Any]:
    """For non-gold-escalate rows a reply must exist; for gold-escalate it may be empty."""
    n = len(golds)
    ok = 0
    for g, r in zip(golds, replies):
        if g == 0 and r.strip():
            ok += 1
        elif g == 1:
            ok += 1
    return {"non_escalate_answered": round(ok / n, 4), "n": n}


def merge_metrics(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> dict[str, Any]:
    gold_esc = [_bin(g["gold_escalate"]) for g in golds]
    pred_esc = [_bin(p["escalate"]) for p in preds]

    sig = binary_metrics(gold_esc, pred_esc)
    sig["metric"] = "escalation"

    int_acc = intent_accuracy([p["intent"] for p in preds], [g["gold_intent"] for g in golds])
    int_acc["metric"] = "intent"

    reply_ok = reply_presence(gold_esc, [p.get("reply", "") or "" for p in preds])
    reply_ok["metric"] = "reply_presence"

    from collections import Counter
    cm = Counter(zip([g["gold_intent"] for g in golds], [p["intent"] for p in preds]))
    top_confusions = sorted(
        ((g, p, c) for (g, p), c in cm.items() if g != p),
        key=lambda t: -t[2],
    )[:12]

    return {
        "n": len(golds),
        "escalation": sig,
        "intent": int_acc,
        "reply_presence": reply_ok,
        "top_intent_confusions": [[g, p, c] for g, p, c in top_confusions],
    }