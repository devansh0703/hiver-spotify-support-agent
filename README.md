# Hiver SDE-Intern Assessment — AI Support Agent for Spotify Care

A locally-built, cloud-backed AI support agent for a brand's Twitter customer
support (demo brand: **Spotify / @SpotifyCares**), trained from the
customer-support-on-twitter (TWCS) dataset and the Banking77 intent dataset,
orchestrated with **LlamaIndex**, retrieved with **Weaviate hybrid search +
MMR + Jina reranking**, embedded with **snowflake-arctic-embed2 (Ollama, local)**
on the ingest side and **Weaviate Cloud** embeddings on the query side, and
powered by **NVIDIA Nemotron** as the LLM.

```
Customer tweet
   │
   ▼
┌─────────────────────────────┐   intent class (json)        ┌────────────────────────────┐
│ LlamaIndex agent (NIM LLM)  │ ────────────────────────────▶ │  Weaviate Cloud  (SpotifyCase)│
│ - classify                  │   hybrid search + MMR +      │  kind="twitter" 11,730 cases  │
│ - retrieve + rerank         │   Jina rerank (cloud vectors)│  kind="banking77" 13,072 rows │
│ - draft reply / route       │ ◀──────────────────────────── │  (ingest vectors: local Ollama)│
└─────────────────────────────┘   Google arctic-embed-l-v2    └────────────────────────────┘
   │
   ├─ reply grounded in historical resolutions (+ rerank similarity)
   └─ escalate? escalation_reason
```

## Reproduce the headline results in under 15 minutes

```bash
pip install --user --break-system-packages -r requirements.txt   # ~2 min
# keys: NVIDIA_API_KEY exported in ~/.bashrc; .env with WEAVIATE_URL,
#       WEAVIATE_API_KEY, JINA_API_KEY; ollama pull snowflake-arctic-embed2:latest

PYTHONPATH=. python3 scripts/agent_test.py                        # live agent, ~1 min
PYTHONPATH=. python3 scripts/run_eval.py --only-report            # metrics from cached rows, <1 min
PYTHONPATH=. python3 scripts/run_eval.py --workers 1 --limit 15   # fresh end-to-end slice, ~5-8 min
```

The third command re-runs agent + baselines + judge on 15 golden examples
and prints metrics for that slice — a genuinely fresh pipeline proof, no
cache. The full 207-example run (`--limit` omitted) takes ~2h and resumes
from checkpoints; its cached output ships in `eval/results/`.

## What it does

Given a customer's support tweet, the agent:

1. **Classifies** intent / sentiment / abuse / pre-flag (Nemotron, JSON mode).
2. **Retrieves** up to 5 most-similar resolved Spotify support cases
   (hybrid BM25+vector, MMR-diversified, Jina-reranked).
3. **Drafts** a reply in Spotify's historical voice, grounded in the
   retrieved resolutions (no invented URLs / amounts / policy).
4. **Routes**: auto-handle vs escalate, with a stated reason.

## Repository layout

```
config.py                 all config (keys, models, paths, thresholds)
src/weaviate_store.py     Weaviate Cloud cxn, ingest (local embeddings), hybrid+MMR+rerank search
src/embed.py              local Ollama embedder (snowflake-arctic-embed2)
src/llm.py                NVIDIA NIM chat wrapper (json-mode, thinking-off, rate throttle)
src/agent.py              SpotifyAgent: classify -> retrieve -> draft -> route
src/orchestrator.py       LlamaIndex AgentRunner / FunctionTool wrapper
src/data_loader.py        parquet/csv -> cleaned cases
scripts/ingest.py         embed + index Spotify cases (kind="twitter")
scripts/ingest_banking77.py  embed Banking77 (kind="banking77")
scripts/agent_probe.py/.agent_test.py/.agent_worker_probe.py  smoke tests
scripts/build_golden_candidates.py  build candidate pool (OOS, never-answered)
scripts/label_golden.py   attach hand labels -> eval/golden_set.jsonl
scripts/run_eval.py       end-to-end golden evaluation (agent + baselines + judge)
scripts/baselines.py      trivial + simple baselines
scripts/eval_metrics.py   automated metrics
scripts/judge.py          LLM-as-judge rubric
eval/                     golden set, results, notes (metrics.json, eval_summary.md)
data/                     raw datasets (see below)
```

## Environment & keys

- Open an NVIDIA account, get an API key; export it in `~/.bashrc`:
  `export NVIDIA_API_KEY=nvapi-...`
- Create a free Weaviate Cloud cluster; put the URL + API key in `.env`
  alongside a free `JINA_API_KEY`:
  ```
  WEAVIATE_URL="https://...weaviate.cloud"
  WEAVIATE_API_KEY="..."
  JINA_API_KEY="jina_..."
  ```
- Install Ollama and pull the embedding model:
  `ollama pull snowflake-arctic-embed2:latest`

Python 3.14, dependencies (system python, no venv — PEP 668):

```bash
pip install --user --break-system-packages -r requirements.txt
```

## Data

| dataset | location | used as |
|---|---|---|
| TWCS twitter (all brands) | `data/twcs/twcs.csv` | raw source |
| Spotify resolved cases | `data/cases/spotify_cases.parquet` (11,730) | retrieval index (`kind="twitter"`) |
| Banking77 | `data/banking77/train.csv` + `test.csv` (13,083 rows) | intent few-shot index (`kind="banking77"`) |

## Index

Both datasets live in the single Weaviate collection `SpotifyCase` (the free
cluster allows exactly one collection); a `kind` property separates them.

```bash
PYTHONPATH=. python3 scripts/ingest_banking77.py     # ~2-3 min, once
PYTHONPATH=. python3 scripts/ingest.py               # full twitter index (long: 11.7k local embeds + upload)
```

Verify counts:

```bash
PYTHONPATH=. python3 -c 'from src import weaviate_store as w; print(w.count_by_kind())'
# {'twitter': 11730, 'banking77': 13072}
```

## Agent smoke test

```bash
PYTHONPATH=. python3 scripts/agent_test.py
```

## Evaluation (the headline result)

Golden set: **207 hand-labelled examples**, all **out-of-sample** — tweets a
customer sent to @spotifycares that the brand **never answered**, so retrieval
can never cheat by copying the archive. Labels: intent (9-way), escalate
bool + reason. Sampling/labelling methodology: `eval/GOLDEN_NOTES.md`.

```bash
PYTHONPATH=. python3 scripts/run_eval.py --workers 1   # full: agent + baselines + judge (~2h, resumable)
PYTHONPATH=. python3 scripts/run_eval.py --only-report  # re-emit metrics from cached rows
```

Outputs land in `eval/results/`:
- `agent_rows.jsonl` — per-example pipeline outputs (resumable cache)
- `judge_rows.jsonl` — LLM-as-judge rubric scores
- `metrics.json` — agent + trivial + simple baselines, judge stats, agreement
- `eval_summary.md` — human-readable summary

All phases checkpoint so a crashed run can be resumed with the same command.
The NIM endpoint rate-limits hard under concurrency, so `--workers 1` is the
reliable setting.

## Headline numbers

Golden set: 207 hand-labelled out-of-support-archive examples (see Evaluation
above). Full detail in `REPORT.md` / `eval/results/metrics.json`.

| Metric | Agent | Simple | Trivial |
|---|---|---|---|
| Escalation precision | 1.000 | 0.077 | 0.0 |
| Escalation recall | 0.789 | 0.105 | 0.0 |
| Escalation F1 | 0.882 | 0.089 | 0.0 |
| Intent accuracy (9-way) | 0.739 | 0.000 | 0.188 |
| Reply presence (non-escalated) | 1.000 | 0.884 | 1.000 |

LLM-as-judge (Nemotron 3-super-120b): overall 4.70/5, 97.1% of examples ≥4;
voice 4.88, groundedness 4.84, helpfulness 3.97. Judge-vs-human agreement on
a 29-row hand-judged subset: 93% exact match on escalation-decision
correctness, κ=0.50 on binarized helpfulness (`eval/human_judged.jsonl`).

Escalation routing is a draft-LLM decision overridden by a deterministic
safety gate (abuse/pre-flag, account-compromise phrases, legal keywords,
billing-dispute patterns); every escalation carries its `escalation_rule`
for auditability. Iteration history (precision 0.43→1.00 after fixing a
substring-keyword bug and adding billing-dispute coverage):
`REPORT.md` §3.