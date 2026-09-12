"""Run the full golden evaluation.

Phases (all checkpointed / resumable):
  1. agent  — src.agent.SpotifyAgent on each golden message (cached by id)
  2. baselines — trivial (canned, never escalate) + simple (retrieval-only)
  3. judge  — LLM-as-judge rubric on the agent's replies
  4. report — metrics.json + eval_summary.md, plus judge-vs-human agreement

Usage:
  PYTHONPATH=/home/devansh/hiver python3 scripts/run_eval.py [--limit N] [--workers W]
  --only-judge        skip phases 1-2 (re-score existing agent rows)
  --judge-model ID    override config.JUDGE_MODEL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from src import weaviate_store as ws
from src.agent import SpotifyAgent
from scripts.baselines import majority_intent, simple_baseline, trivial_baseline
from scripts.eval_metrics import merge_metrics
from scripts.judge import judge_row

RES = config.EVAL_OUT
AGENT_ROW = os.path.join(RES, "agent_rows.jsonl")
BASE_ROW = os.path.join(RES, "baseline_rows.jsonl")
JUDGE_ROW = os.path.join(RES, "judge_rows.jsonl")


def _load_rows(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def _append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite(path: str, rows: dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for _, r in sorted(rows.items(), key=lambda kv: int(kv[0])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_agent(sel: list[dict], workers: int) -> None:
    golds = sel
    cached = _load_rows(AGENT_ROW)
    todo = [g for g in golds if g["id"] not in cached or cached[g["id"]].get("_failed")]
    print(f"[agent] cached={len(cached)}  to_run={len(todo)}")

    agent = SpotifyAgent()
    lock = __import__("threading").Lock()
    done = 0
    fails = 0

    def one(g: dict) -> tuple[str, dict, str | None]:
        for attempt in range(3):
            try:
                r = agent.respond(g["message"])
                return g["id"], r, None
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    return g["id"], {"fatal": str(exc)}, str(exc)
                time.sleep(3 * (attempt + 1))

    for i in range(0, len(todo), 500):
        batch = todo[i : i + 500]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, g) for g in batch]
            for fut in as_completed(futs):
                rid, resp, err = fut.result()
                row = {"id": rid, **resp}
                row["_failed"] = err is not None
                if err is not None:
                    fails += 1
                with lock:
                    _append(AGENT_ROW, row)
                    done += 1
                    if done % 10 == 0:
                        print(f"[agent] {done}/{len(todo)} fails={fails}")
    # refresh cache count for downstreams
    print("[agent] done. rows on disk:", len(_load_rows(AGENT_ROW)))


def run_baselines(sel: list[dict]) -> None:
    golds = sel
    cached = _load_rows(BASE_ROW)
    todo = [g for g in golds if g.get("id") not in cached]
    print(f"[baseline] cached={len(cached)} to_run={len(todo)}")

    maj = majority_intent(golds)
    rows = dict(cached)
    for i, g in enumerate(todo, 1):
        try:
            triv = trivial_baseline(g["message"], maj)
            simp = simple_baseline(g["message"])
        except Exception as exc:  # noqa: BLE001
            triv = {"bl": "trivial", "intent": "unknown", "escalate": False, "reply": "", "_failed": str(exc)}
            simp = {"bl": "simple", "intent": "unknown", "escalate": True, "reply": "", "_failed": str(exc)}
        rows[g["id"]] = {"id": g["id"], "trivial": triv, "simple": simp}
        if i % 10 == 0:
            _rewrite(BASE_ROW, rows)
            print(f"[baseline] {i}/{len(todo)}")
    _rewrite(BASE_ROW, rows)
    print("[baseline] done")


def run_judge(sel: list[dict] | None, model: str | None) -> None:
    agent_rows = _load_rows(AGENT_ROW)
    cached = _load_rows(JUDGE_ROW)
    sel_ids = {g["id"] for g in sel} if sel else None
    todo_ids = [
        rid for rid in agent_rows
        if (sel_ids is None or rid in sel_ids)
        and (rid not in cached or cached[rid].get("judge_error"))
    ]
    print(f"[judge] cached={len(cached)} to_run={len(todo_ids)}")

    lock = __import__("threading").Lock()
    done = 0

    def one(rid: str) -> tuple[str, dict]:
        r = agent_rows[rid]
        try:
            return rid, judge_row(r, model=model)
        except Exception as exc:  # noqa: BLE001
            return rid, {"id": rid, "judge_error": str(exc), "message": r.get("message", "")}

    with ThreadPoolExecutor(max_workers=min(6, 4)) as pool:
        futs = [pool.submit(one, rid) for rid in todo_ids]
        for fut in as_completed(futs):
            rid, row = fut.result()
            row["id"] = rid
            with lock:
                _append(JUDGE_ROW, row)
                done += 1
                if done % 10 == 0:
                    print(f"[judge] {done}/{len(todo_ids)}")
    print("[judge] done. rows on disk:", len(_load_rows(JUDGE_ROW)))


def report(sel: list[dict]) -> None:
    golds = sel
    agent_rows = {g["id"]: _load_rows(AGENT_ROW).get(g["id"]) for g in golds}
    agent_rows = {k: v for k, v in agent_rows.items() if v and not v.get("_failed")}
    base_rows = _load_rows(BASE_ROW)
    judge_rows = _load_rows(JUDGE_ROW)

    usable = [g for g in golds if g["id"] in agent_rows]
    stats = {
        "golden_total": len(golds),
        "agent_ok": len(usable),
        "agent_failed": len(golds) - len(usable),
        "judged": len([r for r in judge_rows.values() if "judge" in r]),
    }

    preds = [agent_rows[g["id"]] for g in usable]
    agg = merge_metrics(preds, usable)
    agg.update(stats)
    agg["mean_intent_confidence"] = round(
        sum(p.get("intent_confidence", 0) or 0 for p in preds) / len(preds), 3
    )
    agg["mean_reply_len"] = round(
        sum(len((p.get("reply") or "")) for p in preds) / len(preds), 1
    )

    # baselines
    baseline_metrics = {}
    for bl in ("trivial", "simple"):
        bl_preds, bl_golds = [], []
        for g in usable:
            row = base_rows.get(g["id"], {})
            bp = row.get(bl) or {}
            if not bp or bp.get("_failed"):
                continue
            bl_preds.append({"intent": bp.get("intent", ""), "escalate": bp.get("escalate", False), "reply": bp.get("reply", "")})
            bl_golds.append(g)
        baseline_metrics[bl] = merge_metrics(bl_preds, bl_golds)
        baseline_metrics[bl]["n"] = len(bl_golds)

    judge_scores = _aggregate_judge(usable, judge_rows)
    agreement = _judge_human_agreement()
    summary = _render(agg, baseline_metrics, judge_scores, agreement)

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump({"agent": agg, "baselines": baseline_metrics, "judge": judge_scores, "agreement": agreement},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(RES, "eval_summary.md"), "w") as f:
        f.write(summary)
    print(summary)


def _aggregate_judge(usable, judge_rows: dict[str, dict]) -> dict:
    scored = [r["judge"] for rid, r in judge_rows.items() if rid in {g["id"] for g in usable} and isinstance(r.get("judge"), dict) and not r.get("judge_error")]
    if not scored:
        return {"n": 0, "note": "no judge rows"}
    from statistics import mean
    agg = {"n": len(scored)}
    for key in ("helpfulness", "voice", "groundedness", "escalation_decision", "overall"):
        vals = [s.get(key) for s in scored if isinstance(s.get(key), (int, float))]
        agg[key] = {"mean": round(mean(vals), 3), "n": len(vals)} if vals else None
    pct = sum(1 for s in scored if (s.get("overall") or 0) >= 4) / len(scored)
    agg["pct_overall_ge4"] = round(pct, 4)
    low = [(s.get("overall"), s.get("notes")) for s in scored if (s.get("overall") or 5) < 3.5]
    agg["low_scoring_notes"] = low[:8]
    return agg


def _kappa(a: list[int], b: list[int]) -> float | None:
    """Cohen's kappa on two binary rating lists."""
    pa = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    p_yes = (sum(a) / len(a)) * (sum(b) / len(b))
    p_no = (1 - sum(a) / len(a)) * (1 - sum(b) / len(b))
    pe = p_yes + p_no
    return round((pa - pe) / (1 - pe), 3) if pe < 1 else None


def _judge_human_agreement() -> dict:
    """Compare the LLM-judge's escalation_decision + helpfulness against a
    human-judged subset (eval/human_judged.jsonl) if present."""
    hp = config.HUMAN_JUDGED_PATH
    if not os.path.exists(hp):
        return {"note": "no human_judged.jsonl yet; add ~15 and rerun"}
    humans = [json.loads(l) for l in open(hp)]
    golds = {json.loads(l)["id"]: json.loads(l) for l in open(config.GOLDEN_PATH)}
    judge_rows = _load_rows(JUDGE_ROW)
    pairs = []
    for h in humans:
        outer = judge_rows.get(h["id"], {})
        j = outer.get("judge")
        if not j or h["id"] not in golds:
            continue
        # agent's actual routing decision lives on the OUTER judge row
        pairs.append((h, j, golds[h["id"]], bool(outer.get("escalate"))))
    if not pairs:
        return {"note": "no matching judge rows"}
    # The judge's escalation_decision is a CORRECTNESS score (5 = the agent's
    # decision was right, 1 = clearly wrong), not a direction vote. Agreement
    # with gold therefore = the judge endorsing decisions on rows where the
    # agent's decision matches gold, plus the judge PENALIZING the rows where
    # the agent diverged from gold.
    agree_rows = [(h, j, g) for h, j, g, ae in pairs if ae == bool(g["gold_escalate"])]
    divergent = [(h, j, g) for h, j, g, ae in pairs if ae != bool(g["gold_escalate"])]
    endorse = sum(1 for _, j, _ in agree_rows if (j.get("escalation_decision") or 0) >= 4) / len(agree_rows)
    div_scores = [(j.get("escalation_decision"), g["id"]) for _, j, g in divergent]
    # judge-vs-human on the SAME reply: both rate decision CORRECTNESS 1-5
    hesc = [h.get("human_escalation") for h, _, _, _ in pairs if isinstance(h.get("human_escalation"), int)]
    jesc = [j.get("escalation_decision") for _, j, _, _ in pairs if isinstance(j.get("escalation_decision"), int)]
    esc_jh = None
    if len(hesc) == len(jesc) and len(hesc) >= 2:
        esc_jh = {
            "n": len(hesc),
            "exact_match": round(sum(1 for a, b in zip(hesc, jesc) if a == b) / len(hesc), 4),
            "mean_abs_diff": round(sum(abs(a - b) for a, b in zip(hesc, jesc)) / len(hesc), 3),
            "note": "human prevalence skew makes kappa degenerate; diff is the informative stat",
        }
    helpful_pairs = [(h["human_helpfulness"], j.get("helpfulness"))
                     for h, j, _, _ in pairs
                     if isinstance(h.get("human_helpfulness"), int)
                     and isinstance(j.get("helpfulness"), (int, float))]
    exact = sum(1 for a, b in helpful_pairs if a == b) / len(helpful_pairs) if helpful_pairs else None
    mad = sum(abs(a - b) for a, b in helpful_pairs) / len(helpful_pairs) if helpful_pairs else None
    hpairs = [(h, j) for h, j, _, _ in pairs
              if isinstance(h.get("human_helpfulness"), int)
              and isinstance(j.get("helpfulness"), (int, float))]
    bina = [1 if h["human_helpfulness"] >= 4 else 0 for h, j in hpairs]
    binb = [1 if j["helpfulness"] >= 4 else 0 for h, j in hpairs]
    k = _kappa(bina, binb) if len(hpairs) >= 2 else None
    return {
        "n": len(pairs),
        "judge_endorses_agent_decision_where_it_matches_gold": round(endorse, 4),
        "n_divergent_from_gold_in_subset": len(divergent),
        "judge_scores_on_divergent_rows": div_scores,
        "escalation_judge_vs_human": esc_jh,
        "helpfulness_exact_match": exact,
        "helpfulness_mean_abs_diff": round(mad, 3) if mad is not None else None,
        "helpfulness_binarized_kappa": k,
    }


def _render(agg, baselines, judge, agreement) -> str:
    def fmt(m):
        e = m["escalation"]
        i = m["intent"]
        return (f"escalation acc={e['accuracy']} prec={e['precision']} rec={e['recall']} "
                f"f1={e['f1']} (tp/fp/fn/tn={e['tp']}/{e['fp']}/{e['fn']}/{e['tn']}, rate={e['escalate_rate']})\n"
                f"intent acc={i['accuracy']}  reply_presence={m['reply_presence']['non_escalate_answered']}")

    lines = ["# Eval summary"]
    lines.append(f"\n**agent** n={agg['agent_ok']} failed={agg['agent_failed']}\n" + fmt(agg))
    for bl, m in baselines.items():
        lines.append(f"\n**{bl}** n={m['n']}\n" + fmt(m))
    if judge.get("n"):
        lines.append(f"\n**judge** n={judge['n']}")
        for k in ("helpfulness", "voice", "groundedness", "escalation_decision", "overall"):
            if judge.get(k):
                lines.append(f"  {k}: mean={judge[k]['mean']} (n={judge[k]['n']})")
        lines.append(f"  pct overall>=4: {judge['pct_overall_ge4']}")
    else:
        lines.append("\n**judge** no rows")
    lines.append("\n**judge-human agreement** " + json.dumps(agreement))
    lines.append("\n**top intent confusions (agent)**")
    for g, p, c in agg.get("top_intent_confusions", []):
        lines.append(f"  gold={g} -> pred={p} ({c})")
    return "\n".join(lines) + "\n"


def _sample_golden(limit: int | None) -> list[dict]:
    """Load the golden set; with --limit, take a STRATIFIED slice covering every
    intent bucket (file order is by intent, so a plain [:limit] would show one
    intent only)."""
    golds = [json.loads(l) for l in open(config.GOLDEN_PATH)]
    if not limit:
        return golds
    by_bucket: dict[str, list[dict]] = {}
    for g in golds:
        by_bucket.setdefault(g["bucket"], []).append(g)
    n_buckets = len(by_bucket)
    per = max(1, limit // n_buckets)
    sel: list[dict] = []
    for b in sorted(by_bucket):  # sorted -> deterministic across runs
        sel.extend(by_bucket[b][:per])
    # top up to exactly `limit` from the untouched remainder
    chosen = {id(g) for g in sel}
    for g in golds:
        if len(sel) >= limit:
            break
        if id(g) not in chosen:
            sel.append(g)
            chosen.add(id(g))
    return sel[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only-judge", action="store_true")
    ap.add_argument("--only-report", action="store_true")
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    sel = _sample_golden(args.limit)

    if not args.only_report:
        ws.wait_until_ready(120)  # only phases that hit Weaviate need it
        if not args.only_judge:
            run_agent(sel, args.workers)
            run_baselines(sel)
        run_judge(sel, args.judge_model)
    report(sel)


if __name__ == "__main__":
    main()