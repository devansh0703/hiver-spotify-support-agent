"""Embed + upload the full Banking77 train/test split into Weaviate Cloud.

Embedding happens LOCALLY via Ollama snowflake-arctic-embed2 (same model as the
TWCS index); only the query-time vectorization is done by Weaviate Cloud.

Rows are stored in the single collection as kind="banking77" with their 77-way
intent in `label`. They are usable for few-shot intent classification only.
"""
from __future__ import annotations

import time

import pandas as pd

import config
from src import weaviate_store as ws
from src.embed import LocalEmbedder, health_check


def load_banking77() -> pd.DataFrame:
    train = pd.read_csv(config.BANKING77_TRAIN, dtype=str)
    test = pd.read_csv(config.BANKING77_TEST, dtype=str)
    df = pd.concat([train, test], ignore_index=True)
    text_col = "text"
    label_col = "category" if "category" in df.columns else "label"
    df = df.rename(columns={label_col: "label"})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() >= 2]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def main() -> None:
    assert health_check(), "Ollama + snowflake-arctic-embed2 must be running"

    t0 = time.time()
    rows = load_banking77()
    print(f"[ingest_banking77] {len(rows)} rows ({rows['label'].nunique()} intents) "
          f"in {time.time()-t0:.1f}s")

    ws.wait_until_ready()
    embedder = LocalEmbedder()

    t0 = time.time()
    stats = ws.ingest(rows, embedder, kind="banking77", wipe=False)
    print(f"[ingest_banking77] upload done in {time.time()-t0:.1f}s: {stats}")
    print(f"[ingest_banking77] total objects: {ws.count()}")
    print(f"[ingest_banking77] per-kind counts: {ws.count_by_kind()}")


if __name__ == "__main__":
    main()