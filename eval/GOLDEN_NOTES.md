# Golden evaluation set — sampling & labelling note

File: `eval/golden_set.jsonl` (207 examples, hand-labelled).

## Why out-of-sample examples

The Weaviate index (`SpotifyCase`) contains 11,730 resolved Spotify tweets. Any
message that the brand already answered — and that a retriever could therefore
surface — would bias evaluation upward: the agent could copy a real answer
rather than reason about the current message. To keep evaluation clean, the
golden set uses messages the index has **never seen**.

## Universe definition

Source: `data/twcs/twcs.csv` (~2.8M tweets across all brands).

Filters applied to build the candidate pool (`scripts/build_golden_candidates.py`):

1. `inbound` = `author_id != brand_author_id` (customer-authored).
2. `text` contains `@spotifycares` (the support brand handle, case-insensitive).
3. `in_response_to_tweet_id` is empty (no explicit reply context).
4. **Never answered**: keep only messages where no later resolved reply from the
   brand references the customer's tweet (resolved replies drawn from
   `data/cases/all_resolved_spotify.parquet`). Guarantees zero overlap with the
   index (whose corpus is exactly the resolved-parquet cases).
5. Deduplicated on normalized message text (`resp.duplicate` column) so repeated
   identical complaints count once.
6. `len(text) >= 12` after cleaning — reject near-empty / URL-only fragments.
7. Drop non-English-like noise by keeping messages with a reasonable letter
   ratio (URLs stripped).

Resulting universe: **17,795 candidate messages** (out of 31,353 inbound
@spotifycares messages; 18,644 were never resolved/answered — some lost to
length/dupe filters).

## Buckets

The agent routes based on a 9-way intent taxonomy (see `config.INTENTS`). To
guarantee coverage of every intent in the 150–250 example target, messages were
stratified: a keyword/pattern classifier mapped each candidate to a bucket,
then **23 examples were sampled per bucket** (207 total) via a deterministic
seeded RNG so selection is reproducible.

Buckets: `technical_issue`, `account_admin`, `billing_subscription`,
`account_access`, `content_library`, `device_integration`,
`feature_request_feedback`, `closing_acknowledgement`, `other_offtopic`.

Because messages are noisy and the bucket heuristic is imperfect, every example
was **hand-relabelled** against the same intent taxonomy; ~1/3 of rows changed
bucket. Final distribution:

```
feature_request_feedback 39   technical_issue 29   billing_subscription 27
account_access 26              content_library 24   other_offtopic 22
closing_acknowledgement 16     account_admin 12     device_integration 12
```

(9 sampled buckets in, 9 intents out — but counts differ because hand labels
override bucket heuristics.)

## Labelling protocol

Each row carries four hand-set fields:

- `gold_intent` — one of the 9 intents, decided from the whole message (plus
  obvious thread context when the message self-references previous turns).
- `gold_escalate` — boolean. Mirrors the agent's escalation policy: escalate
  when the case is abusive/legal, involves a deceased account, a compromised /
  hijacked / unauthorized-access account, a billing dispute / refund on a paid
  plan, or a repeated unresolved troubleshooting attempt. Routine factual
  questions and resolvable issues do **not** escalate.
- `gold_escalation_reason` — free-text justification for the escalation decision
  (empty when `gold_escalate == false`).
- `bucket` — original sampling bucket (kept for provenance).

19/207 (9.2%) are gold-escalate.

`scripts/label_golden.py` attaches the labels; the label map is explicit inside
the script so every decision is auditable against the raw message.

## Reproduce

```bash
PYTHONPATH=/home/devansh/hiver python3 scripts/build_golden_candidates.py
PYTHONPATH=/home/devansh/hiver python3 scripts/label_golden.py
```

Candidate preview with id | bucket | message is at `/tmp/golden_view.txt`.