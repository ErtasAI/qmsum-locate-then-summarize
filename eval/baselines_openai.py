"""Zero-shot GPT baselines under the frozen protocol, cached per query.

Two execution modes, identical request bodies:
  sync  -- one HTTP call per query, full rate, real per-query latency
  batch -- /v1/batches, 50% off input and output, no meaningful latency

Both write the same cache format; the "mode" field on each cached entry decides
which price column that row is billed at. See eval/batch.py.
"""
import argparse, datetime, json, pathlib, time
from data.normalize import load_meetings, load_queries
from eval import batch as B
from eval import protocol
from eval.pricing import cost_usd, load_prices

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Model IDs are floating aliases. The snapshot the API actually served is recorded
# per query as model_snapshot; results rows must carry the snapshot, never the alias.
# See ../../api-model-retirement-ledger.md.
MODELS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
          "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]

# Measured against the live API on 2026-07-27, not assumed:
#   temperature=0 -> 400 "does not support 0 with this model. Only the default (1)
#   value is supported" on every GPT-5.6 tier; accepted on every GPT-5.4 tier.
# So the frozen protocol's TEMPERATURE cannot be honoured on 5.6. Greedy decoding
# is simply not expressible there, exactly as on Claude 5. Report it, don't hide it.
NO_TEMPERATURE = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}

# The 5.6 tier reasons by default. Left alone, reasoning tokens are billed as output
# AND consume max_completion_tokens, so a 512-token cap would be spent thinking and
# return an empty summary. reasoning_effort="none" is the supported off switch
# ('minimal' is rejected; supported values are none/low/...).
REASONING_MODELS = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}

BATCH_ENDPOINT = "/v1/chat/completions"

# Enqueued-token ceilings, taken verbatim from the vendor's own rejection on
# 2026-07-27: "Enqueued token limit reached for <model>. Limit: N enqueued tokens."
# These are ORG-TIER limits for this account, not universal constants, and OpenAI
# raises them with usage tier. The QMSum test split is ~3.5M tokens, so EVERY model
# here needs the split submitted in chunks, each collected before the next is
# enqueued. Anthropic imposed no equivalent limit; its 281-request batch went
# through whole.
ENQUEUED_TOKEN_LIMIT = {
    "gpt-5.6-sol": 900_000,
    "gpt-5.6-luna": 2_000_000,
    "gpt-5.6-terra": 900_000,     # not measured; assume the stricter observed value
}
DEFAULT_ENQUEUED_LIMIT = 900_000
ENQUEUE_SAFETY = 0.85

# Measured 4.75 chars/token on QMSum prompts (59,934 chars -> 12,630 real tokens).
# 4.0 is deliberately conservative: over-estimating tokens shrinks the chunk, which
# is the safe direction for a hard limit that fails the whole batch when crossed.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(prompt):
    return len(prompt) / CHARS_PER_TOKEN + protocol.MAX_NEW_TOKENS


def build_prompt(query, utterances, budget=None):
    transcript = protocol.truncate_words(protocol.format_transcript(utterances),
                                         budget or protocol.PROMPT_WORD_BUDGET)
    return protocol.PROMPT_TEMPLATE.format(query=query, transcript=transcript)


def request_body(model, prompt, thinking="off"):
    """The scored request. Shared by sync and batch so the two are comparable."""
    body = {"model": model,
            "max_completion_tokens": protocol.MAX_NEW_TOKENS,
            "messages": [{"role": "user", "content": prompt}]}
    if model not in NO_TEMPERATURE:
        body["temperature"] = protocol.TEMPERATURE
    if model in REASONING_MODELS:
        body["reasoning_effort"] = "none" if thinking == "off" else thinking
    if thinking != "off":
        # reasoning is billed against max_completion_tokens; the answer needs headroom
        body["max_completion_tokens"] = protocol.MAX_NEW_TOKENS * 8
    return body


def cache_entry(body_or_resp, mode, budget, thinking="off", latency_s=None):
    """Normalise a chat-completion payload (SDK object or batch dict) into cache form.

    reasoning_tokens are recorded separately for analysis but are ALREADY counted
    inside completion_tokens by OpenAI's accounting, so they must not be added again
    when costing the row.
    """
    if isinstance(body_or_resp, dict):
        text = body_or_resp["choices"][0]["message"]["content"]
        usage, snapshot = body_or_resp["usage"], body_or_resp["model"]
        pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        finish = body_or_resp["choices"][0].get("finish_reason")
    else:
        r = body_or_resp
        text, snapshot = r.choices[0].message.content, r.model
        pt, ct = r.usage.prompt_tokens, r.usage.completion_tokens
        rt = getattr(getattr(r.usage, "completion_tokens_details", None), "reasoning_tokens", None)
        finish = r.choices[0].finish_reason
    return {"prediction": text, "prompt_tokens": pt, "completion_tokens": ct,
            "reasoning_tokens": rt, "finish_reason": finish,
            "latency_s": latency_s, "model_snapshot": snapshot,
            "called_at": B.utcnow(), "prompt_word_budget": budget,
            "thinking": thinking, "mode": mode}


# --- synchronous ------------------------------------------------------------

def run_split(client, model, split, limit=None, prices=None, cache_dir=None, preds_dir=None,
              budget=None, thinking="off"):
    prices = prices or load_prices()
    budget = budget or protocol.PROMPT_WORD_BUDGET
    tag = B.run_tag(model, budget, thinking)
    cache_dir = pathlib.Path(cache_dir or B.cache_dir_for(tag))
    preds_dir = pathlib.Path(preds_dir or ROOT / "results" / "preds")
    cache_dir.mkdir(parents=True, exist_ok=True); preds_dir.mkdir(parents=True, exist_ok=True)
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    queries = load_queries(split)[:limit]
    out = preds_dir / f"baseline-{tag}-{split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for q in queries:
            cache = cache_dir / f"{q['query_id']}.json"
            if cache.exists():
                c = json.loads(cache.read_text(encoding="utf-8"))
            else:
                t0 = time.time()
                resp = client.chat.completions.create(
                    **request_body(model, build_prompt(q["query"], meetings[q["meeting_id"]], budget),
                                   thinking))
                c = cache_entry(resp, "sync", budget, thinking, round(time.time() - t0, 3))
                cache.write_text(json.dumps(c), encoding="utf-8")
            f.write(json.dumps(B.preds_row(model, q["query_id"], c, prices)) + "\n")
    return out


# --- batch ------------------------------------------------------------------

def submit_batch(client, model, split, limit=None, budget=None, thinking="off",
                 cache_dir=None, jobs_dir=None, max_enqueued_tokens=None):
    """Enqueue as many missing queries as the model's enqueued-token ceiling allows.

    Returns None when the cache is already complete, so a resubmit is a no-op rather
    than a rebill. When the split does not fit in one batch, this submits the first
    chunk and reports `remaining`; collect it, then call again to send the next.
    """
    budget = budget or protocol.PROMPT_WORD_BUDGET
    tag = B.run_tag(model, budget, thinking)
    cache_dir = pathlib.Path(cache_dir or B.cache_dir_for(tag))
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = B.pending(split, cache_dir, limit)
    if not todo:
        return None
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    cap = (max_enqueued_tokens or ENQUEUED_TOKEN_LIMIT.get(model, DEFAULT_ENQUEUED_LIMIT))
    cap *= ENQUEUE_SAFETY
    lines, est_tokens = [], 0.0
    for q in todo:
        prompt = build_prompt(q["query"], meetings[q["meeting_id"]], budget)
        est = estimate_tokens(prompt)
        if lines and est_tokens + est > cap:     # always send at least one request
            break
        lines.append(json.dumps({"custom_id": q["query_id"], "method": "POST",
                                 "url": BATCH_ENDPOINT,
                                 "body": request_body(model, prompt, thinking)}))
        est_tokens += est
    payload = "\n".join(lines).encode("utf-8")
    up = client.files.create(file=(f"{tag}-{split}.jsonl", payload), purpose="batch")
    b = client.batches.create(input_file_id=up.id, endpoint=BATCH_ENDPOINT,
                              completion_window="24h",
                              metadata={"tag": tag, "split": split})
    job = {"provider": "openai", "batch_id": b.id, "input_file_id": up.id,
           "model": model, "split": split, "tag": tag, "budget": budget,
           "thinking": thinking,
           "n_requests": len(lines), "remaining": len(todo) - len(lines),
           "est_enqueued_tokens": int(est_tokens), "enqueued_cap": int(cap),
           "payload_mb": round(len(payload) / 1e6, 2),
           "submitted_at": B.utcnow(), "status": b.status,
           "cache_dir": str(cache_dir)}
    B.save_job(job, jobs_dir)
    return job


def poll_batch(client, job):
    b = client.batches.retrieve(job["batch_id"])
    counts = getattr(b, "request_counts", None)
    return {"status": b.status,
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
            "total": getattr(counts, "total", None),
            "output_file_id": b.output_file_id,
            "error_file_id": b.error_file_id}


def collect_batch(client, job, cache_dir=None):
    """Write every succeeded response into the per-query cache. Errors are
    returned, not cached, so a rerun retries exactly the failures."""
    state = poll_batch(client, job)
    if state["status"] != "completed":
        return state, 0, []
    cache_dir = pathlib.Path(cache_dir or job.get("cache_dir") or B.cache_dir_for(job["tag"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    written, errors = 0, []
    text = client.files.content(state["output_file_id"]).text
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        resp = rec.get("response") or {}
        if rec.get("error") or resp.get("status_code") != 200:
            errors.append((rec.get("custom_id"), rec.get("error") or resp.get("status_code")))
            continue
        c = cache_entry(resp["body"], "batch", job["budget"], job.get("thinking", "off"))
        (cache_dir / f"{rec['custom_id']}.json").write_text(json.dumps(c), encoding="utf-8")
        written += 1
    if state.get("error_file_id"):
        for line in client.files.content(state["error_file_id"]).text.splitlines():
            if line.strip():
                rec = json.loads(line)
                errors.append((rec.get("custom_id"), rec.get("error")))
    return state, written, errors


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    from openai import OpenAI
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache-dir", help="override the cache location (use for throwaway smoke tests)")
    ap.add_argument("--budget", type=int, default=protocol.PROMPT_WORD_BUDGET,
                    help="transcript word budget; anything other than the protocol default "
                         "is a DEVIATION and is tagged into the run name")
    ap.add_argument("--thinking", choices=["off", "low", "medium", "high"], default="off",
                    help="off = reasoning_effort=none, matches the Claude rows (default). "
                         "Anything else is the reasoning regime, tagged as a separate run.")
    a = ap.parse_args()
    out = run_split(OpenAI(), a.model, a.split, limit=a.limit, budget=a.budget,
                    cache_dir=a.cache_dir, thinking=a.thinking)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
