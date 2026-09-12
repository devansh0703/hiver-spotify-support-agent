"""Spotify support agent.

Message -> [classify] -> [retrieve grounded history] -> [draft + route]

The agent runs the whole Twitter-style pipeline: intent classification,
sentiment, a grounded reply draft built from how Spotify historically
resolved similar issues, and an auto-handle vs escalate decision with a
stated reason.
"""
from __future__ import annotations

import json
import re
from typing import Any

import config
from src import weaviate_store as ws
from src.llm import chat, extract_json

CLASSIFIER_PROMPT = """You are the classifier for {brand} customer support on Twitter.

Customer message:
{message}

Prior turns in this conversation (oldest first; empty if none):
{prior_turns}

Classify the LATEST customer message. Respond ONLY with JSON:
{{
  "intent": one of {intents},
  "confidence": 0.0-1.0,
  "sentiment": "very_negative"|"negative"|"neutral"|"positive",
  "account_identifiers": true if the message includes an email address, order/account number, phone number, or other account-identifying info,
  "abusive": true ONLY if the message contains direct insults, hostile abuse, obscene gestures, or profanity AIMED AT the brand, its employees, or its agents. Frustration, sarcasm, passing swearing, or complaints about the product are NOT abusive (false). Examples: "your app is dumb af" = false; "you people are useless morons" = true; "🖕" = true;
  "pre_flagged": true if the message mentions legal action, threats, or demands to speak to a human
}}

Intent definitions:
- technical_issue: bug, crash, playback failure, offline/connectivity, downloads, freeze, malfunction.
- account_admin: merge accounts, change email/username/country, delete account, artist page, privacy settings.
- billing_subscription: plans, charges, refunds, discounts, promos, cancellation of paid plan, gift cards.
- account_access: cannot sign in, password reset, locked/hacked, logged out unexpectedly.
- content_library: missing/unavailable songs/albums/artists, greyed-out tracks, region availability, playlists, library, local files, radio, search.
- device_integration: using Spotify on car/TV/speakers/watch/browser-desktop, Chromecast, connecting devices.
- feature_request_feedback: feature requests, suggestions, opinion/feedback/praise, no malfunction.
- closing_acknowledgement: thanks, "done", confirmations, farewell. No new issue.
- other_offtopic: spam, off-topic chatter, memes, abusive/troll content.
"""

DRAFT_PROMPT = """You are {brand} social care on Twitter. Draft a reply to a customer.

LATEST customer message:
{message}

Prior turns (oldest first; empty if none):
{prior_turns}

Classified: intent={intent} (conf {confidence}), sentiment={sentiment},
abusive={abusive}, pre_flagged={pre_flagged}

Historical resolutions retrieved from Spotify's own support archive
(most similar customer issues first, with similarity of the customer issue):
{resolutions}

Rules:
- Mirror how Spotify historically resolved THIS KIND of issue (the resolution
  examples above). Follow their structure and next step, but rewrite in your
  own words; never copy verbatim.
- Voice: warm, concise, human, lowercase-and-affordable; 1-3 short sentences;
  at most one emoji and NEVER slash-codes like /SJ, /G, /HEART.
  Spell out contractions (we're, we'll).
- Never invent URLs, links, refund amounts, compensation, or policy. If the
  resolution requires account access (DM/email) state what info you need.
- If the message is abusive, pre-flagged, or unresolvable without account
  access you MUST NOT draft a reply — leave "reply" empty and escalate.

Route decision:
Auto-handle when (a) the issue is routine, (b) similarity to a well-resolved
historical case is high (rerank >= 0.5), and (c) it can be answered without
access to the customer's account data or 3rd parties.
Escalate when: abusive/pre-flagged, legal, account compromised, billing
dispute on a paid plan, multi-message failure ("still doesn't work"), or the
best-matching historical resolution has low similarity (rerank < 0.35).

Respond ONLY with JSON:
{{
  "escalate": true|false,
  "escalation_reason": "short why (empty if not escalating)",
  "auto_reply_ready": true|false,
  "reply": "the draft reply text (empty if escalate)",
  "grounded_in": "tweet id(s) of the historical resolutions used, or 'none'"
}}

Use the exact tweet ids from the resolutions list, e.g. "reply_12732".
"""


def _fmt_resolutions(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "(no similar historical cases found)"
    lines = []
    for i, c in enumerate(cases, 1):
        rs = c.get("rerank_score")
        rs = f"{rs:.3f}" if rs is not None else "n/a"
        lines.append(
            f"{i}. CUSTOMER issue: {c['text'][:220]}\n"
            f"   CUSTOMER tweet_id: {c['customer_tweet_id']}\n"
            f"   BRAND historically replied (rerank {rs}): {c['reply'][:500]}"
        )
    return "\n".join(lines)


def _esc_gate(message: str, clf: dict, draft: dict) -> tuple[bool, str, str]:
    """Deterministic escalation override — safety-critical signals always win.

    Returns (escalate, reason, rule) where rule names which signal fired.
    """
    text = message.lower()
    if clf.get("abusive") or clf.get("pre_flagged"):
        return True, "abusive or legally pre-flagged message", "abuse_preflag"
    compromise_kw = [
        "hacked", "compromised", "took over", "stolen", "unauthorized",
        "did not request", "didn't request", "not me", "someone else",
        "unknown devices", "controlling my account", "my email was changed",
        "email was changed", "login was changed", "password was changed",
    ]
    for kw in compromise_kw:
        if kw in text:
            return True, f"possible account compromise or unauthorized access", "compromise"
    # config keywords: WORD-BOUNDARY only — substring matching made for example
    # "sue" fire on "issue" because ESCALATION_KEYWORDS includes "sue".
    for kw in config.ESCALATION_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return True, f"escalation keyword: {kw}", "keyword"
    billing_dispute = re.compile(
        r"\brefund\b"
        r"|\bchargeback\b"
        r"|never signed up"
        r"|full (amount|price|charge)"
        r"|charged (full|9\.9\d?)"
        r"|money (was )?(has been )?(been )?taken"
        r"|not applied to my account"
        r"|charged (me )?\d+(\.\d+)?.*?(but|instead|then|now|not)"
    )
    if billing_dispute.search(text):
        return True, "billing dispute or refund request on a paid plan", "billing_dispute"
    return bool(draft.get("escalate")), draft.get("escalation_reason", ""), "llm"


class SpotifyAgent:
    def __init__(self, model: str | None = None, top_k: int | None = None) -> None:
        self.model = model or config.AGENT_MODEL
        self.top_k = top_k or config.RERANK_TOP
        self.intent_list = ", ".join(config.INTENTS)

    def respond(
        self,
        message: str,
        prior_turns: list[str] | None = None,
        n_retrieval: int | None = None,
    ) -> dict[str, Any]:
        prior_turns = prior_turns or []
        prior_block = "\n".join(f"- {t[:180]}" for t in prior_turns[-6:]) or "(none)"

        # --- 1) classify ---
        classifier_prompt = CLASSIFIER_PROMPT.format(
            brand=config.BRAND_NAME,
            message=message,
            prior_turns=prior_block,
            intents=self.intent_list,
        )
        clf_raw = chat(
            [{"role": "user", "content": classifier_prompt}],
            model=self.model,
            max_tokens=4096,
            json_mode=True,
        )
        clf = extract_json(clf_raw)

        # --- 2) retrieve grounded history ---
        n_retrieval = n_retrieval or self.top_k
        cases = ws.search(message, limit_k=n_retrieval)

        # --- 3) draft + route ---
        draft_prompt = DRAFT_PROMPT.format(
            brand=config.BRAND_NAME,
            message=message,
            prior_turns=prior_block,
            intent=clf.get("intent", "unknown"),
            confidence=clf.get("confidence", 0.0),
            sentiment=clf.get("sentiment", "neutral"),
            abusive=clf.get("abusive", False),
            pre_flagged=clf.get("pre_flagged", False),
            resolutions=_fmt_resolutions(cases),
        )
        draft_raw = chat(
            [{"role": "user", "content": draft_prompt}],
            model=self.model,
            max_tokens=4096,
            json_mode=True,
        )
        draft = extract_json(draft_raw)

        escalate, esc_reason, esc_rule = _esc_gate(message, clf, draft)
        reply = "" if escalate else draft.get("reply", "").strip()
        # Nemotron sometimes appends social-care-ish slash-codes ("/gk", "/TB");
        # strip trailing artifacts deterministically.
        reply = re.sub(r"\s*\/[A-Za-z0-9]{1,4}\s*$", "", reply)

        return {
            "message": message,
            "intent": clf.get("intent", "unknown"),
            "intent_confidence": clf.get("confidence", 0.0),
            "sentiment": clf.get("sentiment", "neutral"),
            "account_identifiers": clf.get("account_identifiers", False),
            "abusive": clf.get("abusive", False),
            "pre_flagged": clf.get("pre_flagged", False),
            "escalate": escalate,
            "escalation_reason": esc_reason,
            "escalation_rule": esc_rule,
            "draft_escalate": bool(draft.get("escalate")),
            "auto_reply_ready": draft.get("auto_reply_ready", False),
            "reply": reply,
            "grounded_in": draft.get("grounded_in", "none"),
            "retrieved": [
                {
                    "text": c["text"],
                    "reply": c["reply"],
                    "rerank_score": c.get("rerank_score"),
                    "customer_tweet_id": c.get("customer_tweet_id"),
                }
                for c in cases
            ],
        }


if __name__ == "__main__":
    agent = SpotifyAgent()
    r = json.dumps(agent.respond("my app keeps crashing every time I open a playlist"), indent=2)
    print(r[:3000])