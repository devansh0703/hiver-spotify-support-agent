"""Sample the golden eval set candidates from unresolved inbound Spotify tweets.

Sampling design (see eval/GOLDEN_NOTES.md):
  * Universe: 18,644 customer tweets that @-mention SpotifyCares, are inbound,
    were never answered by the brand (so there is ZERO overlap with the Weaviate
    retrieval index -> no leak), and have >= 12 cleaned chars.
  * Stratification: coarse keyword-ish buckets approximate the 9 intents so
    every intent class gets judged, then simple random sampling within buckets.
  * Output: eval/golden_candidates.jsonl (message + bucket + source tweet id);
    labels are then added by hand to produce eval/golden_set.jsonl.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from src.data_loader import clean_text

RNG = pd.np.random if hasattr(pd, "np") else __import__("numpy").random.default_rng(42)

BUCKETS: dict[str, str] = {
    "technical_issue": "crash|glitch|bug|freez|buffering|not working|audio|volume|sound|quality|downloaded|download|offline|sync|silence|skip|slow|error|activation|connect|wifi|data",
    "account_admin": "merge|merge account|username|email address|change .*email|country|delete account|deactivate|artist|privacy|profile|dummy|duplicate",
    "billing_subscription": "charged|charge|refund|premium|subscription|plan|billing|payment|card|discount|student|family|gift|cancel|renew|promo|free trial|debit|invoice",
    "account_access": "login|log in|sign in|password|hacked|locked|forgot|verify|email verification|spam|breach|unauthorized|logged out",
    "content_library": "song|album|artist|playlist|lyrics|radio|podcast|missing|unavailable|remove|grey|greyed|search|library|local files|shuffle|region",
    "device_integration": "car|android auto|carplay|tv|sonos|speaker|echo|alexa|watch|chromecast|ps5|xbox|desktop|mobile|windows|mac|ps4|fire",
    "feature_request_feedback": "please add|feature|suggestion|recommend|would love|idea|vote|feedback|wish|could you|request|praise|love|thanks for",
    "closing_acknowledgement": "thanks|thank you|ty|thx|done|worked|appreciate|great|fixed",
    "other_offtopic": "",
}


def bucket_of(text: str) -> str:
    t = text.lower()
    # explicit greed order: a message about password reset is access, not admin.
    if re.search(BUCKETS["account_access"], t):
        return "account_access"
    if re.search(BUCKETS["billing_subscription"], t):
        return "billing_subscription"
    for name, pat in BUCKETS.items():
        if name in ("account_access", "billing_subscription"):
            continue
        if pat and re.search(pat, t):
            return name
    return "other_offtopic"


def build() -> None:
    os.makedirs("eval", exist_ok=True)
    df = pd.read_csv("data/twcs/twcs.csv", dtype=str, low_memory=False, on_bad_lines="skip")
    tids = set(
        df[(df["author_id"] == "SpotifyCares") & (df["inbound"] == "False")][
            "response_tweet_id"
        ].str.strip().str.replace(".0", "", regex=False).dropna()
    )
    cust = df[df["inbound"] == "True"].copy()
    cust["clean"] = cust["text"].map(clean_text)
    mentioned = cust[cust["text"].str.contains("@spotifycares", case=False, na=False)]
    universe = mentioned[~mentioned["tweet_id"].isin(tids)]
    universe = universe[universe["clean"].str.len() >= 12]
    universe = universe.drop_duplicates(subset="clean").copy()
    universe["bucket"] = universe["clean"].map(bucket_of)
    print("universe:", len(universe), "| bucket dist:")
    print(universe["bucket"].value_counts().to_string())

    # target 200; give a little headroom poor-buckets
    per = max(8, round(210 / universe["bucket"].nunique()))
    rng = __import__("numpy").random.default_rng(7)
    picked = []
    for b, grp in universe.groupby("bucket"):
        n = min(len(grp), per)
        grp = grp.sample(n=n, random_state=rng)
        for _, row in grp.iterrows():
            picked.append(
                {
                    "id": row["tweet_id"],
                    "bucket": b,
                    "message": row["clean"],
                    "raw": row["text"],
                }
            )
    with open("eval/golden_candidates.jsonl", "w") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("wrote", len(picked), "candidates -> eval/golden_candidates.jsonl")


if __name__ == "__main__":
    build()