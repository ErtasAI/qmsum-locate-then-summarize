"""Zero-shot Claude baselines under the frozen protocol, cached per query.

Mirrors baselines_openai.py. Three protocol notes, all of which belong in the paper:

1. Claude 5 models REJECT `temperature` (400). Greedy decoding is not expressible
   on them, so the temperature-0 half of the frozen protocol cannot be honoured for
   claude-opus-5 / claude-sonnet-5. Haiku 4.5 still accepts it and gets 0.0.
2. Thinking is ON by default on the Claude 5 models. We disable it so these rows sit
   in the same non-reasoning regime as the GPT rows. `--thinking low` runs the
   reasoning regime instead, as a separate tagged run.
3. With thinking disabled, Opus 5 can leak internal XML into the visible response,
   which would corrupt ROUGE. A system-level instruction suppresses it and
   scan_leaks() checks the predictions afterwards. Never score a run that leaked.

Two execution modes with identical request params: sync, and the Message Batches
API at 50% off. Unlike the OpenAI batch path, results come back as typed Message
objects, so both modes share one parser.
"""
import argparse, json, pathlib, re, time
from data.normalize import load_meetings, load_queries
from eval import batch as B
from eval import protocol
from eval.pricing import load_prices

ROOT = pathlib.Path(__file__).resolve().parents[1]

MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
NO_SAMPLING_PARAMS = {"claude-opus-5", "claude-sonnet-5"}   # temperature/top_p/top_k -> 400
THINKING_ON_BY_DEFAULT = {"claude-opus-5", "claude-sonnet-5"}

# Kept out of the frozen user prompt so the scored prompt stays byte-identical
# across providers; this rides in the system field instead.
ANTI_LEAK_SYSTEM = "Do not include internal or system XML tags in your response."
LEAK_RE = re.compile(r"</?(thinking|antml|internal|scratchpad)\b", re.I)


def build_prompt(query, utterances, budget=None):
    transcript = protocol.truncate_words(protocol.format_transcript(utterances),
                                         budget or protocol.PROMPT_WORD_BUDGET)
    return protocol.PROMPT_TEMPLATE.format(query=query, transcript=transcript)


def request_kwargs(model, thinking):
    """Per-model request shape. See the module docstring for why these differ."""
    kw = {"max_tokens": protocol.MAX_NEW_TOKENS}
    if thinking == "off":
        if model in THINKING_ON_BY_DEFAULT:
            kw["thinking"] = {"type": "disabled"}   # accepted at effort high or below
        if model not in NO_SAMPLING_PARAMS:
            kw["temperature"] = protocol.TEMPERATURE
        kw["system"] = ANTI_LEAK_SYSTEM
    else:
        if model in THINKING_ON_BY_DEFAULT:
            kw["thinking"] = {"type": "adaptive"}
            kw["output_config"] = {"effort": "low"}
        # thinking is billed against max_tokens, so the answer needs its own headroom
        kw["max_tokens"] = protocol.MAX_NEW_TOKENS * 8
    return kw


def extract_text(resp):
    return "".join(b.text for b in resp.content if b.type == "text")


def cache_entry(resp, mode, budget, thinking, latency_s=None):
    return {"prediction": extract_text(resp),
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "latency_s": latency_s,
            "model_snapshot": resp.model,
            "called_at": B.utcnow(),
            "stop_reason": resp.stop_reason,
            "prompt_word_budget": budget,
            "thinking": thinking,
            "mode": mode}


def scan_leaks(path):
    leaked = [json.loads(l)["query_id"] for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
              if l.strip() and LEAK_RE.search(json.loads(l)["prediction"] or "")]
    return leaked


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
    kwargs = request_kwargs(model, thinking)
    out = preds_dir / f"baseline-{tag}-{split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for q in queries:
            cache = cache_dir / f"{q['query_id']}.json"
            if cache.exists():
                c = json.loads(cache.read_text(encoding="utf-8"))
            else:
                t0 = time.time()
                resp = client.messages.create(
                    model=model,
                    messages=[{"role": "user",
                               "content": build_prompt(q["query"], meetings[q["meeting_id"]], budget)}],
                    **kwargs)
                if resp.stop_reason == "refusal":
                    raise RuntimeError(f"{model} refused on {q['query_id']}: {resp.stop_details}")
                c = cache_entry(resp, "sync", budget, thinking, round(time.time() - t0, 3))
                cache.write_text(json.dumps(c), encoding="utf-8")
            f.write(json.dumps(B.preds_row(model, q["query_id"], c, prices)) + "\n")
    return out


# --- batch ------------------------------------------------------------------

def submit_batch(client, model, split, limit=None, budget=None, thinking="off",
                 cache_dir=None, jobs_dir=None):
    """Submit the missing queries as one Message Batch. Returns None if the cache
    is already complete, so a resubmit is a no-op rather than a rebill."""
    budget = budget or protocol.PROMPT_WORD_BUDGET
    tag = B.run_tag(model, budget, thinking)
    cache_dir = pathlib.Path(cache_dir or B.cache_dir_for(tag))
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = B.pending(split, cache_dir, limit)
    if not todo:
        return None
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(split)}
    kwargs = request_kwargs(model, thinking)
    requests = [{"custom_id": q["query_id"],
                 "params": {"model": model,
                            "messages": [{"role": "user",
                                          "content": build_prompt(q["query"],
                                                                  meetings[q["meeting_id"]], budget)}],
                            **kwargs}}
                for q in todo]
    payload_mb = round(len(json.dumps(requests).encode("utf-8")) / 1e6, 2)
    b = client.messages.batches.create(requests=requests)
    job = {"provider": "anthropic", "batch_id": b.id, "model": model, "split": split,
           "tag": tag, "budget": budget, "thinking": thinking,
           "n_requests": len(todo), "payload_mb": payload_mb,
           "submitted_at": B.utcnow(), "status": b.processing_status,
           "cache_dir": str(cache_dir)}
    B.save_job(job, jobs_dir)
    return job


def poll_batch(client, job):
    b = client.messages.batches.retrieve(job["batch_id"])
    rc = getattr(b, "request_counts", None)
    return {"status": b.processing_status,
            "completed": getattr(rc, "succeeded", None),
            "failed": (getattr(rc, "errored", 0) or 0) + (getattr(rc, "expired", 0) or 0),
            "total": job["n_requests"]}


def collect_batch(client, job, cache_dir=None):
    """Cache every succeeded message. Errors are returned, not cached, so a
    resubmit retries exactly the failures."""
    state = poll_batch(client, job)
    if state["status"] != "ended":
        return state, 0, []
    cache_dir = pathlib.Path(cache_dir or job.get("cache_dir") or B.cache_dir_for(job["tag"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    written, errors = 0, []
    for entry in client.messages.batches.results(job["batch_id"]):
        res = entry.result
        if res.type != "succeeded":
            errors.append((entry.custom_id, res.type))
            continue
        msg = res.message
        if msg.stop_reason == "refusal":
            errors.append((entry.custom_id, "refusal"))
            continue
        c = cache_entry(msg, "batch", job["budget"], job.get("thinking", "off"))
        (cache_dir / f"{entry.custom_id}.json").write_text(json.dumps(c), encoding="utf-8")
        written += 1
    return state, written, errors


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    import anthropic
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache-dir", help="override the cache location (use for throwaway smoke tests)")
    ap.add_argument("--budget", type=int, default=protocol.PROMPT_WORD_BUDGET,
                    help="transcript word budget; anything other than the protocol default "
                         "is a DEVIATION and is tagged into the run name")
    ap.add_argument("--thinking", choices=["off", "low"], default="off",
                    help="off = non-reasoning regime, matches the GPT rows (default). "
                         "low = adaptive thinking at low effort, tagged as a separate run.")
    a = ap.parse_args()
    out = run_split(anthropic.Anthropic(), a.model, a.split, limit=a.limit,
                    budget=a.budget, thinking=a.thinking, cache_dir=a.cache_dir)
    leaked = scan_leaks(out)
    print(f"wrote {out}")
    if leaked:
        print(f"WARNING: internal tags leaked into {len(leaked)} prediction(s): {leaked[:5]}")
        print("Do NOT score this run. Re-run with --thinking low.")
    else:
        print("leak scan clean")


if __name__ == "__main__":
    main()
