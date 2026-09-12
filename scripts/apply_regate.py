"""Apply the v2 escalation gate + stricter abuse classifier to cached rows.

The v1 eval ran with (a) substring keyword matching ("sue" fired on "issue")
and (b) a classifier prompt that flagged mild profanity as abusive. Both were
fixed in src/agent.py AFTER that run. Because the gate is deterministic and
the classifier change can only reduce positives, the faithful delta is:

  1. re-classify ONLY the rows the old prompt flagged abusive (8);
  2. re-apply the v2 gate to every row (deterministic, no LLM);
  3. re-draft (LLM) every row whose routing flipped and now auto-handles —
     the sue-FPs and any abuse rows the tightened prompt un-flagged — using
     its fresh classification, then re-apply the gate to the new draft.
     Newly ESCALATED rows (billing disputes) just drop the reply — no LLM;
  4. rewrite agent_rows.jsonl in place. Judge re-scoring of changed rows is
     handled by run_eval.py (rows missing from the judge cache are re-run).

Usage: PYTHONPATH=. python3 scripts/apply_regate.py
"""
from __future__ import annotations

import json
import re

import config
from src.agent import CLASSIFIER_PROMPT, DRAFT_PROMPT, _esc_gate, _fmt_resolutions
from src.llm import chat, extract_json

SLASH_CODE = re.compile(r"\s*\/[A-Za-z0-9]{1,4}\s*$")


def _classify(message: str) -> dict:
    prompt = CLASSIFIER_PROMPT.format(
        brand=config.BRAND_NAME,
        message=message,
        prior_turns="(none)",
        intents=", ".join(config.INTENTS),
    )
    return extract_json(chat(
        [{"role": "user", "content": prompt}],
        model=config.AGENT_MODEL, max_tokens=4096, json_mode=True,
    ))


def _draft_and_gate(row: dict) -> None:
    """Re-run the draft stage with fresh clf, then re-apply the v2 gate."""
    prompt = DRAFT_PROMPT.format(
        brand=config.BRAND_NAME,
        message=row["message"],
        prior_turns="(none)",
        intent=row.get("intent", "unknown"),
        confidence=row.get("intent_confidence", 0.0),
        sentiment=row.get("sentiment", "neutral"),
        abusive=row.get("abusive", False),
        pre_flagged=row.get("pre_flagged", False),
        resolutions=_fmt_resolutions(row.get("retrieved") or []),
    )
    draft = extract_json(chat(
        [{"role": "user", "content": prompt}],
        model=config.AGENT_MODEL, max_tokens=4096, json_mode=True,
    ))
    clf = {"abusive": row.get("abusive", False), "pre_flagged": row.get("pre_flagged", False)}
    escalate, reason, rule = _esc_gate(row["message"], clf, draft)
    row["draft_escalate"] = bool(draft.get("escalate"))
    row["auto_reply_ready"] = draft.get("auto_reply_ready", False)
    row["grounded_in"] = draft.get("grounded_in", "none")
    row["escalate"], row["escalation_reason"], row["escalation_rule"] = escalate, reason, rule
    reply = "" if escalate else (draft.get("reply") or "").strip()
    row["reply"] = SLASH_CODE.sub("", reply)


def main() -> None:
    path = "eval/results/agent_rows.jsonl"
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(path)}

    # --- 1) re-classify rows the OLD prompt flagged abusive ---
    abusive_ids = [rid for rid, r in rows.items() if r.get("abusive") and not r.get("_failed")]
    print(f"re-classifying {len(abusive_ids)} abusive-flagged rows with the tightened prompt")
    unflagged = set()
    for rid in abusive_ids:
        clf = _classify(rows[rid]["message"])
        rows[rid]["intent"] = clf.get("intent", rows[rid].get("intent"))
        rows[rid]["intent_confidence"] = clf.get("confidence", rows[rid].get("intent_confidence"))
        rows[rid]["sentiment"] = clf.get("sentiment", rows[rid].get("sentiment"))
        rows[rid]["account_identifiers"] = clf.get("account_identifiers", False)
        rows[rid]["abusive"] = clf.get("abusive", False)
        rows[rid]["pre_flagged"] = clf.get("pre_flagged", False)
        if not (clf.get("abusive") or clf.get("pre_flagged")):
            unflagged.add(rid)
        print(f"  {rid}: abusive={clf.get('abusive')} pre_flagged={clf.get('pre_flagged')}")

    # --- 2) deterministic v2 gate over every unchanged row ---
    redraft_ids: set[str] = set(unflagged)  # abuse rows that lost their flag -> fresh draft
    for rid, r in rows.items():
        if r.get("_failed") or rid in redraft_ids:
            continue
        if r.get("escalate"):
            # recover the draft decision: gate-forced reasons have fixed prefixes,
            # anything else means the draft LLM itself escalated
            reason = r.get("escalation_reason", "")
            gate_forced = reason.startswith((
                "abusive or legally pre-flagged",
                "possible account compromise",
                "escalation keyword:",
            ))
            draft = ({"escalate": False, "escalation_reason": ""} if gate_forced
                     else {"escalate": True, "escalation_reason": reason})
        else:
            draft = {"escalate": False, "escalation_reason": ""}
        clf = {"abusive": r.get("abusive", False), "pre_flagged": r.get("pre_flagged", False)}
        new_esc, new_reason, new_rule = _esc_gate(r["message"], clf, draft)
        old_esc = bool(r.get("escalate"))
        r["escalate"], r["escalation_reason"], r["escalation_rule"] = new_esc, new_reason, new_rule
        if new_esc != old_esc:
            if old_esc and not new_esc:
                redraft_ids.add(rid)  # v1 FP dropped -> needs a real reply
            print(f"  flip {rid}: {'esc->auto' if old_esc else 'auto->esc'} ({new_rule})")

    # --- 3) LLM: fresh draft + gate for rows that now auto-handle ---
    print(f"re-drafting {len(redraft_ids)} rows: {sorted(redraft_ids, key=int)}")
    for rid in sorted(redraft_ids, key=int):
        _draft_and_gate(rows[rid])
        r = rows[rid]
        tag = "ESC" if r["escalate"] else "auto"
        print(f"  {rid}: {tag} rule={r['escalation_rule']} reply={r['reply'][:55]!r}")

    # --- 4) rewrite in place ---
    with open(path, "w") as f:
        for rid in sorted(rows, key=int):
            f.write(json.dumps(rows[rid], ensure_ascii=False) + "\n")
    print(f"rewrote {path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
