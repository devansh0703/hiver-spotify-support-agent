"""Re-apply the (v2) escalation gate over cached agent rows.

The v2 gate in src/agent.py is deterministic given (message, clf flags,
draft decision). The cached run was produced with the v1 gate; only the
gate changed since, so instead of a blind full re-run we:

  1. re-derive (escalate, reason, rule) for every cached row with the v2 gate;
  2. report which rows flip (dropped FP escalations vs new escalations).

Rows that UN-escalate (the "sue"-in-"issue" FPs) had their reply discarded
by v1, so their reply must be re-drafted by the LLM — handled separately by
scripts/apply_regate.py. This script only DRY-RUNS the flip analysis.

Usage: PYTHONPATH=. python3 scripts/regate.py
"""
from __future__ import annotations

import json

from src.agent import _esc_gate


def _recover_draft(r: dict) -> dict:
    """v1 rows predate escalation_rule, so recover the draft decision from the
    stored escalation_reason: gate-forced reasons have fixed prefixes; anything
    else means the draft LLM itself escalated."""
    if not r.get("escalate"):
        return {"escalate": False, "escalation_reason": ""}
    reason = r.get("escalation_reason", "")
    gate_prefixes = (
        "abusive or legally pre-flagged",
        "possible account compromise",
        "escalation keyword:",
    )
    if reason.startswith(gate_prefixes):
        return {"escalate": False, "escalation_reason": ""}
    return {"escalate": True, "escalation_reason": reason}


def main() -> None:
    rows = [json.loads(l) for l in open("eval/results/agent_rows.jsonl")]
    flips = []
    for r in rows:
        if r.get("_failed"):
            continue
        draft = _recover_draft(r)
        clf = {"abusive": r.get("abusive", False), "pre_flagged": r.get("pre_flagged", False)}
        new_esc, new_reason, new_rule = _esc_gate(r["message"], clf, draft)
        old_esc = bool(r["escalate"])
        if new_esc != old_esc:
            flips.append({"id": r["id"], "old": old_esc, "new": new_esc,
                          "old_reason": r.get("escalation_reason", ""),
                          "new_rule": new_rule, "reason": new_reason})

    dropped = [f for f in flips if f["old"] and not f["new"]]
    added = [f for f in flips if not f["old"] and f["new"]]
    print(f"rows: {len(rows)}   flips: {len(flips)}")
    print(f"\n-- UN-escalated (v1 FPs dropped): {len(dropped)}")
    for f in dropped:
        print(f"  {f['id']}  was: {f['old_reason'][:50]!r}")
    print(f"\n-- NOW escalated (v2 forces): {len(added)}")
    for f in added:
        print(f"  {f['id']}  -> {f['new_rule']} ({f['reason'][:50]})")

    with open("eval/results/regate_flips.json", "w") as fh:
        json.dump({"dropped": dropped, "added": added}, fh, indent=2)
    print("\nwrote eval/results/regate_flips.json")


if __name__ == "__main__":
    main()
