"""Local embedding via Ollama snowflake-arctic-embed2.

All corpus vectors are computed HERE (locally) and later pushed to Weaviate
as explicit vectors, so no cloud embedding credits are spent on ingestion.
"""
from __future__ import annotations

import time

import requests

import config


class LocalEmbedder:
    def __init__(self) -> None:
        self.url = config.OLLAMA_URL.rstrip("/") + "/api/embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), config.OLLAMA_BATCH):
            batch = texts[i : i + config.OLLAMA_BATCH]
            payload = {
                "model": config.OLLAMA_EMBED_MODEL,
                "input": batch,
            }
            for attempt in range(5):
                try:
                    r = requests.post(self.url, json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    emb = data["embeddings"]
                    for e in emb:
                        if len(e) != config.EMBED_DIM:
                            raise ValueError(
                                f"dim {len(e)} != expected {config.EMBED_DIM}"
                            )
                    out.extend(emb)
                    break
                except requests.RequestException as exc:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def health_check() -> bool:
    try:
        r = requests.get(config.OLLAMA_URL.rstrip("/") + "/api/tags", timeout=10)
        tags = [m.get("name") for m in r.json().get("models", [])]
        return any("arctic-embed2" in t for t in tags)
    except requests.RequestException:
        return False


if __name__ == "__main__":
    assert health_check(), "Ollama / arctic-embed2 not reachable"
    e = LocalEmbedder()
    v = e.embed_one("hello world")
    print("ok, dim", len(v))