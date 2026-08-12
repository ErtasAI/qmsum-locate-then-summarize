"""Synchronous probe: observed service latency, and whether batching changes output.

Answers two questions the batched full-split runs cannot.

1. LATENCY. A batch response has no per-query latency. This re-runs a seeded
   subsample synchronously and records it. The result is an OBSERVED SERVICE
   LATENCY for a given model, date and region, NOT a model speed: it includes the
   vendor's queue depth and fleet load at the moment of the call. Adjacent calls to
   the same model have been seen to differ by 2.4x. Report it with n, date and the
   caveat, or do not report it.

2. DOES BATCHING CHANGE QUALITY? It should not, but "should not" is not a
   measurement, and a naive batch-vs-sync text diff cannot answer it because
   GPT-5.6 and Claude 5 reject temperature=0 and run at vendor-default sampling.
   Two sync runs of those models already differ. So the probe needs a control:

     - claude-haiku-4-5 accepts temperature=0. It is the ONLY model in the set
       where sampling noise is removed, making it the decisive case. If its batch
       and sync outputs agree, batching does not change the computation.
     - For a sampling model, --repeat 2 runs the same config twice synchronously to
       establish the NOISE FLOOR. Batch-vs-sync divergence is only evidence of a
       batch effect if it exceeds that floor.

Writes to its own cache and preds directories so it never touches, re-bills, or
overwrites the committed full-split runs. Costs real money: sync rate, no discount.
"""
import argparse, json, pathlib, random, statistics as st, time

from data.normalize import load_meetings, load_queries
from eval import batch as B
from eval import baselines_anthropic as ANTHROPIC
from eval import baselines_openai as OPENAI
from eval import protocol
from eval.pricing import load_prices
from eval.scorer import score

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "results" / "sync_probe"


def pick_queries(split, n):
    """Seeded sample spread across the split. The first n queries would all come
    from the first meeting or two, biasing prompt length and therefore latency."""
    qs = load_queries(split)
    return sorted(random.Random(protocol.SEED).sample(qs, min(n, len(qs))),
                  key=lambda q: q["query_id"])


def run_once(client, provider, model, split, queries, meetings, run_id, prices):
    """One synchronous pass over the chosen queries. Fresh cache dir per run_id so
    a repeat actually re-calls the API instead of reading the first run back."""
    cache = PROBE_DIR / "cache" / run_id
    cache.mkdir(parents=True, exist_ok=True)
    mod = OPENAI if provider == "openai" else ANTHROPIC
    rows = []
    for q in queries:
        cf = cache / f"{q['query_id']}.json"
        if cf.exists():
            c = json.loads(cf.read_text(encoding="utf-8"))
        else:
            prompt = mod.build_prompt(q["query"], meetings[q["meeting_id"]])
            t0 = time.perf_counter()
            if provider == "openai":
                resp = client.chat.completions.create(**mod.request_body(model, prompt))
                c = mod.cache_entry(resp, "sync", protocol.PROMPT_WORD_BUDGET, "off",
                                    round(time.perf_counter() - t0, 3))
            else:
                resp = client.messages.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    **mod.request_kwargs(model, "off"))
                c = mod.cache_entry(resp, "sync", protocol.PROMPT_WORD_BUDGET, "off",
                                    round(time.perf_counter() - t0, 3))
            cf.write_text(json.dumps(c), encoding="utf-8")
        rows.append(B.preds_row(model, q["query_id"], c, prices))
    return rows


def agreement(a, b):
    """How much two prediction sets agree: exact-match rate plus ROUGE of one
    against the other. ROUGE-to-reference on n=30 is too noisy to detect a small
    effect; self-similarity is far more sensitive."""
    pa = [r["prediction"] or "" for r in a]
    pb = [r["prediction"] or "" for r in b]
    exact = sum(x == y for x, y in zip(pa, pb)) / len(pa)
    s = score(pa, pb, with_bertscore=False)
    return {"exact_match_rate": round(exact, 4),
            "self_rouge1": round(s["rouge1"], 4), "self_rougeL": round(s["rougeL"], 4)}


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=1,
                    help="sync passes per model; 2 establishes the sampling noise floor")
    a = ap.parse_args()

    prices = load_prices()
    queries = pick_queries(a.split, a.n)
    meetings = {m["meeting_id"]: m["utterances"] for m in load_meetings(a.split)}
    refs = {q["query_id"]: q["reference"] for q in load_queries(a.split)}
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    clients, report = {}, {}

    for model in a.models:
        provider = "openai" if model in OPENAI.MODELS else "anthropic"
        if provider not in clients:
            if provider == "openai":
                from openai import OpenAI; clients["openai"] = OpenAI(timeout=600)
            else:
                import anthropic; clients["anthropic"] = anthropic.Anthropic(timeout=600)

        passes = [run_once(clients[provider], provider, model, a.split, queries,
                           meetings, f"{model}-r{i}", prices)
                  for i in range(a.repeat)]
        sync = passes[0]
        lat = [r["latency_s"] for r in sync if r.get("latency_s") is not None]

        # the committed batched predictions for exactly these queries
        bpath = ROOT / "results" / "preds" / f"baseline-{model}-{a.split}.jsonl"
        bmap = {json.loads(l)["query_id"]: json.loads(l)
                for l in bpath.read_text(encoding="utf-8").splitlines() if l.strip()}
        batched = [bmap[q["query_id"]] for q in queries]

        entry = {
            "n": len(queries),
            "temperature_0_honoured": (model not in OPENAI.NO_TEMPERATURE
                                       and model not in ANTHROPIC.NO_SAMPLING_PARAMS),
            "latency_s": {"mean": round(st.mean(lat), 3), "median": round(st.median(lat), 3),
                          "min": round(min(lat), 3), "max": round(max(lat), 3),
                          "spread_ratio": round(max(lat) / min(lat), 2)},
            "sync_cost_usd": round(sum(r["cost_usd"] for r in sync), 4),
            "batch_vs_sync": agreement(batched, sync),
            "rouge1_batch": round(score([r["prediction"] or "" for r in batched],
                                        [refs[r["query_id"]] for r in batched],
                                        with_bertscore=False)["rouge1"], 4),
            "rouge1_sync": round(score([r["prediction"] or "" for r in sync],
                                       [refs[r["query_id"]] for r in sync],
                                       with_bertscore=False)["rouge1"], 4),
        }
        if a.repeat > 1:
            entry["sync_vs_sync_NOISE_FLOOR"] = agreement(passes[0], passes[1])
        report[model] = entry
        print(f"[{model}] {json.dumps(entry)}")

    # MERGE, never clobber: the output name is keyed on split and n, so invoking the
    # probe once per model group (cheap models with --repeat 2 for the noise floor,
    # expensive ones with --repeat 1) would otherwise have the second run silently
    # delete the first run's results, which is exactly what happened on 2026-07-27.
    out = PROBE_DIR / f"probe-{a.split}-n{a.n}.json"
    merged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    merged.update(report)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nwrote {out} ({len(merged)} models: {', '.join(sorted(merged))})")


if __name__ == "__main__":
    main()
