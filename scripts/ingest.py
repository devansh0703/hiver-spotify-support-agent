"""Step 1: build case dataset, embed locally, upload to Weaviate Cloud."""
from __future__ import annotations

import time

from src import weaviate_store as ws
from src.data_loader import build
from src.embed import health_check, LocalEmbedder


def main() -> None:
    assert health_check(), "Ollama + snowflake-arctic-embed2 must be running"

    t0 = time.time()
    cases = build()
    print(f"[ingest] {len(cases)} cases built in {time.time()-t0:.1f}s")

    ws.wait_until_ready()
    embedder = LocalEmbedder()

    t0 = time.time()
    stats = ws.ingest(cases, embedder, kind="twitter", wipe=True)
    print(f"[ingest] upload done in {time.time()-t0:.1f}s: {stats}")
    print(f"[ingest] total objects in collection: {ws.count()}")
    print(f"[ingest] per-kind counts: {ws.count_by_kind()}")


if __name__ == "__main__":
    main()