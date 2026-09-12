"""Weaviate Cloud store: collection lifecycle, local-embedding ingest, hybrid+MMR+rerank query.

Vector-space contract:
  * Index — every case is embedded LOCALLY (Ollama snowflake-arctic-embed2)
    and uploaded as an explicit vector. The collection still declares the
    `text2vec-weaviate` vectorizer (`Snowflake/snowflake-arctic-embed-l-v2.0`)
    so that, at QUERY time, Weaviate Cloud generates the query embedding for us.
    The two are the same model family (1024 dims), so the forward/query vectors
    live in the same space.
  * Query — raw text in, no vector sent; the cloud vectorizer embeds the query;
    hybrid search fuses BM25 + the cloud vector; MMR diversifies; the Jina AI
    reranker re-orders on raw text.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import weaviate
from weaviate.auth import Auth
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Diversity, Filter, Rerank

import config
from src.embed import LocalEmbedder


def connect() -> weaviate.WeaviateClient:
    headers = {}
    if config.JINA_API_KEY:
        headers["X-JinaAI-Api-Key"] = config.JINA_API_KEY
    # skip_init_checks: the gRPC health ping runs early and can fail on some
    # networks even though data operations over gRPC/REST work fine.
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=Auth.api_key(config.WEAVIATE_API_KEY),
        headers=headers,
        skip_init_checks=True,
    )


def ensure_collection(client: weaviate.WeaviateClient, wipe: bool = False) -> None:
    if wipe and client.collections.exists(config.COLLECTION):
        client.collections.delete(config.COLLECTION)
        print(f"[weaviate_store] wiped existing {config.COLLECTION}")
    created = False
    if not client.collections.exists(config.COLLECTION):
        client.collections.create(
            config.COLLECTION,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="reply", data_type=DataType.TEXT, index_searchable=False),
                Property(name="customer_tweet_id", data_type=DataType.TEXT),
                Property(name="reply_tweet_id", data_type=DataType.TEXT),
                Property(name="customer_id", data_type=DataType.TEXT),
                Property(name="kind", data_type=DataType.TEXT),
                Property(name="label", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.text2vec_weaviate(
                name="embedding",
                source_properties=["text"],
                model=config.WEAVIATE_CLOUD_MODEL,
            ),
            reranker_config=Configure.Reranker.jinaai(
                model="jina-reranker-v2-base-multilingual"
            ),
        )
        created = True
        print(f"[weaviate_store] created collection {config.COLLECTION}")

    col = client.collections.get(config.COLLECTION)
    have = {p.name for p in col.config.get(simple=True).properties}
    for prop in ("kind", "label"):
        if prop not in have:
            col.config.add_property(
                Property(name=prop, data_type=DataType.TEXT, index_filterable=True)
            )
            print(f"[weaviate_store] added property {prop} to {config.COLLECTION}")
    return created


def ingest(
    cases: pd.DataFrame,
    embedder: LocalEmbedder,
    kind: str = "twitter",
    wipe: bool = False,
) -> dict[str, int]:
    """Upload rows with locally computed vectors.

    Args:
        kind: discriminator stored on every object. "twitter" rows carry a
            customer tweet + historical brand reply; "banking77" rows carry a
            query + label (intent) and are only used for intent few-shot work.
        wipe: if True, delete the whole collection first (only for a full
            re-index). Otherwise the collection is added to.
    """
    client = connect()
    ensure_collection(client, wipe=wipe)
    collection = client.collections.get(config.COLLECTION)

    texts = cases["text"].tolist()
    vectors = embedder.embed(texts)

    failed = 0
    ok = 0
    batch_size = 100
    for i in range(0, len(cases), batch_size):
        chunk = cases.iloc[i : i + batch_size]
        vec_chunk = vectors[i : i + batch_size]
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for (_, row), vec in zip(chunk.iterrows(), vec_chunk):
                props: dict[str, Any] = {
                    "text": row["text"],
                    "kind": kind,
                }
                if kind == "twitter":
                    props.update(
                        {
                            "reply": row["reply"],
                            "customer_tweet_id": str(row["customer_tweet_id"]),
                            "reply_tweet_id": str(row["reply_tweet_id"]),
                            "customer_id": str(row["customer_id"]),
                        }
                    )
                elif "label" in cases.columns:
                    props["label"] = str(row["label"])
                batch.add_object(
                    properties=props,
                    vector={"embedding": vec},
                    uuid=None,
                )
            n_errors = batch.number_errors
            failed += n_errors
            ok += len(chunk) - n_errors
        print(f"[weaviate_store] {i + len(chunk)}/{len(cases)} inserted (kind={kind}, ok={ok} fail={failed})")
    client.close()
    return {"ok": ok, "failed": failed}


def backfill_kind(kind: str = "twitter") -> int:
    """Set `kind` on existing objects that don't carry it (pre-kind schema)."""
    client = connect()
    collection = client.collections.get(config.COLLECTION)
    filt = Filter.by_property("kind").is_none()
    updated = 0
    after: str | None = None
    while True:
        objs = collection.query.fetch_objects(
            after=after,
            limit=500,
            filters=filt,
            return_properties=["kind"],
        )
        batch = objs.objects
        if not batch:
            break
        for obj in batch:
            collection.data.update(
                uuid=obj.uuid,
                properties={"kind": kind},
            )
            updated += 1
        after = str(batch[-1].uuid)
        print(f"[weaviate_store] backfilled {updated} objects -> kind={kind}")
    client.close()
    return updated


def search(
    message: str,
    limit_k: int | None = None,
    kind: str = "twitter",
) -> list[dict[str, Any]]:
    """Hybrid (BM25 + cloud-vector) search, then MMR, then Jina rerank.

    Returns the top `RERANK_TOP` cases as dicts with text/reply/scores.
    kind="twitter" retrieves resolved support cases for reply drafting;
    kind="banking77" retrieves labelled intents for few-shot classification.
    """
    message = message.strip()
    limit_k = limit_k or config.RERANK_TOP
    client = connect()
    collection = client.collections.get(config.COLLECTION)

    response = collection.query.hybrid(
        query=message,
        alpha=config.HYBRID_ALPHA,
        limit=config.MMR_CANDIDATES,
        filters=Filter.by_property("kind").equal(kind),
        diversity_selection=Diversity.mmr(limit=config.MMR_RESULTS, balance=config.MMR_BALANCE),
        rerank=Rerank(prop="text", query=message),
        return_metadata=["score"],
        return_properties=["text", "reply", "customer_tweet_id", "customer_id", "kind", "label"],
    )
    client.close()

    results = []
    for obj in response.objects:
        rerank_score = getattr(obj.metadata, "rerank_score", None)
        results.append(
            {
                "text": obj.properties["text"],
                **({} if obj.properties.get("reply") is None else {"reply": obj.properties["reply"]}),
                "customer_tweet_id": obj.properties.get("customer_tweet_id"),
                "customer_id": obj.properties.get("customer_id"),
                "kind": obj.properties.get("kind"),
                "label": obj.properties.get("label"),
                "hybrid_score": obj.metadata.score,
                "rerank_score": rerank_score,
            }
        )
    return results[:limit_k]


def count_by_kind() -> dict[str, int]:
    client = connect()
    counts = {}
    for kind in ("twitter", "banking77"):
        counts[kind] = (
            client.collections.get(config.COLLECTION)
            .aggregate.over_all(total_count=True, filters=Filter.by_property("kind").equal(kind))
            .total_count
        )
    client.close()
    return counts


def count() -> int:
    client = connect()
    n = (
        client.collections.get(config.COLLECTION)
        .aggregate.over_all(total_count=True)
        .total_count
    )
    client.close()
    return n


def wait_until_ready(timeout_s: int = 60) -> None:
    client = connect()
    for _ in range(timeout_s):
        try:
            if client.is_ready():
                client.close()
                return
        except Exception:  # noqa: S110
            pass
        time.sleep(1)
    client.close()
    raise TimeoutError("Weaviate cluster not ready")


if __name__ == "__main__":
    wait_until_ready()
    print("count:", count())