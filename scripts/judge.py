"""LLM-as-judge reply quality scoring with a 1-5 rubric (NVIDIA NIM).

Also used for judge-vs-human agreement: the same rubric is applied by a human
on a subset (eval/human_judged.jsonl) and the scores are compared.
"""
from __future__ import annotations

from typing import Any

import config
from src.llm import chat, extract_json

RUBRIC_PROMPT = """You are a strict but fair QA judge for {brand} social care on Twitter.

Customer message:
{message}

Classified intent: {intent}
Escalation decision made by the system: {escalate}. Reason: {reason}

Historical resolutions the reply was built from (top retrieved, most similar first):
{resolutions}

Candidate reply from the system:
{reply}

Rate the candidate on FOUR 1-5 scales:
- helpfulness: 5 = correct, actionable next step that would resolve the issue; 3 = partial; 1 = unhelpful, off-topic, or a non-sequitur.
- voice:        5 = warm, concise, human, on-brand; 3 = stiff/windy; 1 = robotic, cold, or hostile.
- groundedness: 5 = fully faithful to the historical resolutions, no invented details; 3 = mixes supported and invented; 1 = fabricated URLs, amounts, or policy.
                If the reply is empty (escalated), leave groundedness as null.
- escalation_decision: 5 = clearly correct (escalates real abuse/legal/compromise/billing-dispute; auto-handles routine asks); 3 = borderline either way; 1 = clearly wrong (auto-replies to abuse/legal/compromise, or escalates a trivial question).

Respond ONLY with JSON:
{{
  "helpfulness": 1-5 or null if reply empty,
  "voice": 1-5 or null if reply empty,
  "groundedness": 1-5 or null if reply empty,
  "escalation_decision": 1-5,
  "overall": rounded 1-decimal mean of the non-null scores,
  "notes": "one short sentence"
}}
"""


def judge_row(row: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """Score one predicted instance. Returns judge JSON plus the inputs for audit."""
    model = model or config.JUDGE_MODEL
    reply = (row.get("reply") or "").strip()
    resolutions = row.get("retrieved") or row.get("from_retrieval") or []
    res_lines = []
    for i, c in enumerate(resolutions[:5], 1):
        r = c.get("rerank_score")
        r = f"{r:.3f}" if r is not None else "n/a"
        res_lines.append(
            f"{i}. (rerank {r}) ISSUE: {str(c.get('text', ''))[:180]}\n"
            f"   RESOLUTION: {str(c.get('reply', ''))[:300]}"
        )
    res_block = "\n".join(res_lines) or "(none retrieved)"

    prompt = RUBRIC_PROMPT.format(
        brand=config.BRAND_NAME,
        message=row.get("message", ""),
        intent=row.get("intent", "unknown"),
        escalate=str(bool(row.get("escalate"))),
        reason=row.get("escalation_reason", "") or "n/a",
        resolutions=res_block,
        reply=reply or "(empty — escalated, no draft)",
    )
    raw = chat([{"role": "user", "content": prompt}], model=model, max_tokens=1000, json_mode=True)
    parsed = extract_json(raw)
    return {
        "model": model,
        "message": row.get("message", ""),
        "reply": reply,
        "escalate": bool(row.get("escalate")),
        "judge": {
            "helpfulness": parsed.get("helpfulness"),
            "voice": parsed.get("voice"),
            "groundedness": parsed.get("groundedness"),
            "escalation_decision": parsed.get("escalation_decision"),
            "overall": parsed.get("overall"),
            "notes": parsed.get("notes", ""),
        },
        "rubric_input": prompt,
    }