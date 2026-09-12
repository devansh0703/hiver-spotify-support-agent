"""Load the TWCS CSV and build the consolidated Spotify case dataset.

Each case = one customer inbound tweet + the brand's historical reply,
which is the unit both retrieval and reply-drafting ground on.
"""
from __future__ import annotations

import re

import pandas as pd

import config

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")


def clean_text(s: str | float) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = _URL_RE.sub(" ", s)
    s = s.replace("&amp;", "&")
    s = _MENTION_RE.sub(" ", s)
    s = s.replace("#", "")
    s = _WS_RE.sub(" ", s)
    s = s.strip(" .")
    return s.strip()


def load_cases(max_cases: int | None = None) -> pd.DataFrame:
    """Return the case frame for the configured brand."""
    df = pd.read_csv(config.RAW_CSV, dtype=str, low_memory=False, on_bad_lines="skip")

    brand_out = df[
        (df["author_id"] == config.BRAND_HANDLE) & (df["inbound"] == "False")
    ].copy()
    brand_out["response_tweet_id"] = brand_out["response_tweet_id"].str.strip()

    reply_targets = brand_out["response_tweet_id"].dropna()
    cust = df[df["tweet_id"].isin(reply_targets)].copy()

    # inner join: customer tweet -> its Spotify reply
    merged = cust.merge(
        brand_out[["response_tweet_id", "text", "tweet_id", "created_at"]],
        left_on="tweet_id",
        right_on="response_tweet_id",
        suffixes=("_cust", "_reply"),
    )

    cases = pd.DataFrame(
        {
            "text": merged["text_cust"].map(clean_text),
            "reply": merged["text_reply"].map(clean_text),
            "customer_id": merged["author_id"],
            "customer_tweet_id": merged["tweet_id_cust"],
            "reply_tweet_id": merged["tweet_id_reply"],
            "created_at": merged["created_at_reply"],
        }
    )

    cases = cases[cases["text"].str.len() >= 5]
    cases = cases[cases["reply"].str.split().str.len() >= config.MIN_REPLY_TOKENS]
    cases = cases.drop_duplicates(subset=["text"])

    if max_cases is not None:
        cases = cases.head(max_cases).copy()

    return cases.reset_index(drop=True)


def build() -> pd.DataFrame:
    import os

    os.makedirs(config.CASES_DIR, exist_ok=True)
    cases = load_cases(max_cases=config.MAX_CASES)
    cases.to_parquet(config.CASES_PATH, index=False)
    print(f"[data_loader] wrote {len(cases)} cases -> {config.CASES_PATH}")
    return cases


if __name__ == "__main__":
    build()