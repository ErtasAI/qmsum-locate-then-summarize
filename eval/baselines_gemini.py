"""Zero-shot Gemini baselines under the frozen protocol, cached per query.

Mirrors baselines_openai.py / baselines_anthropic.py. Two notes:

1. COST IS LOGGED AT LIST PRICE even when the run is served by free-tier quota.
   A $0.00 row would make the paper's cost-per-quality table meaningless; the
   number a reader needs is what this workload costs a buyer, not what it cost us.
   Free quota changes our bill, not the economics being compared.
2. Gemini Flash models think by default. `thinking_budget=0` puts them in the same
   non-reasoning regime as the GPT and Claude rows. `--thinking` runs the reasoning
   regime instead, as a separately tagged run.

Unlike the OpenAI and Anthropic paths, this API accepts a seed, so these rows are
the only baselines that are reproducible in the strict sense.
"""
import argparse, json, pathlib, time
from data.normalize import load_meetings, load_queries
from eval import batch as B
from eval import protocol
from eval.pricing import load_prices

ROOT = pathlib.Path(__file__).resolve().parents[1]

MODELS = ["gemini-3.6-flash", "gemini-3.5-flash"]

def build_prompt(query, utterances, budget=None):
    transcript = protocol.truncate_words(protocol.format_transcript(utterances),
                                         budget or protocol.PROMPT_WORD_BUDGET)
    return protocol.PROMPT_TEMPLATE.format(query=query, transcript=transcript)

# Gemini 3.x replaced thinking_budget with thinking_level, and MINIMAL is the floor:
# thinking_budget=0 returns 400 INVALID_ARGUMENT on gemini-3.6-flash (measured
# 2026-07-27). Thinking cannot be switched off on this family at all. The Gemini rows
# are therefore NOT in the same strictly-non-reasoning regime as the GPT rows
# (reasoning_effort="none") or the Claude rows (thinking disabled). That asymmetry is
# a property of the vendors' APIs, not a choice; the paper states it rather than
# implying a like-for-like decoding regime across all three.
THINKING_FLOOR = "MINIMAL"

# Observed against the live endpoint on 2026-07-27 while running the test split.
# Retrying these is correct: they clear on their own within seconds to minutes.
TRANSIENT = ("UNAVAILABLE", "503", "INTERNAL", "500", "DEADLINE_EXCEEDED", "504",
             "RESOURCE_EXHAUSTED", "429")

# ...but NOT every 429 is transient. The free tier enforces a PER-DAY cap as well as
# a per-minute one, and they arrive as the same status code. The daily one is
# identified by quotaId "...PerDay..." and does not clear for hours, so retrying it
# burns the full backoff chain (~10 min) per query and still fails. Measured free-tier
# daily cap on gemini-3.6-flash: 20 requests. A 281-query split cannot be run on it.
HARD_QUOTA = ("PerDay", "GenerateRequestsPerDayPerProjectPerModel")

class DailyQuotaExhausted(RuntimeError):
    """Free-tier daily request cap hit. Not retryable; needs billing or a new day."""

def call_with_backoff(fn, tries=8, base=5, sleep=time.sleep):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if any(h in msg for h in HARD_QUOTA):
                raise DailyQuotaExhausted(
                    "Gemini free-tier DAILY request quota exhausted (measured cap: 20/day "
                    "on gemini-3.6-flash). This is not a rate limit that backoff can clear. "
                    "Enable billing on the Google AI Studio project to run the full split, "
                    "or drop the Gemini rows. Cached responses are kept; a rerun resumes."
                ) from e
            if not any(t in msg for t in TRANSIENT) or i == tries - 1:
                raise
            sleep(min(base * (2 ** i), 300))


def build_config(thinking):
    from google.genai import types
    kw = {"temperature": protocol.TEMPERATURE,
          "max_output_tokens": protocol.MAX_NEW_TOKENS,
          "seed": protocol.SEED}
    if thinking == "off":
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=THINKING_FLOOR)
    return types.GenerateContentConfig(**kw)

def run_split(client, model, split, limit=None, prices=None, cache_dir=None, preds_dir=None,
              budget=None, thinking="off", config=None):
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
                c = json.loads(cache.read_text())
            else:
                t0 = time.time()
                resp = call_with_backoff(lambda: client.models.generate_content(
                    model=model,
                    contents=build_prompt(q["query"], meetings[q["meeting_id"]], budget),
                    config=config if config is not None else build_config(thinking)))
                u = resp.usage_metadata
                c = {"prediction": resp.text,
                     "prompt_tokens": u.prompt_token_count,
                     "completion_tokens": u.candidates_token_count or 0,
                     "thoughts_tokens": getattr(u, "thoughts_token_count", None) or 0,
                     "latency_s": round(time.time() - t0, 3),
                     "model_snapshot": resp.model_version,
                     "called_at": B.utcnow(),
                     "prompt_word_budget": budget,
                     "thinking": thinking,
                     "mode": "sync"}
                cache.write_text(json.dumps(c), encoding="utf-8")
            # preds_row folds thoughts_tokens into billed output; omitting them would
            # understate cost, since Gemini bills thinking as output tokens
            f.write(json.dumps(B.preds_row(model, q["query_id"], c, prices)) + "\n")
    return out

def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    from google import genai
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache-dir", help="override the cache location (use for throwaway smoke tests)")
    ap.add_argument("--budget", type=int, default=protocol.PROMPT_WORD_BUDGET,
                    help="transcript word budget; anything other than the protocol default "
                         "is a DEVIATION and is tagged into the run name")
    ap.add_argument("--thinking", choices=["off", "on"], default="off",
                    help="off = non-reasoning regime, matches the GPT and Claude rows (default)")
    a = ap.parse_args()
    out = run_split(genai.Client(), a.model, a.split, limit=a.limit,
                    budget=a.budget, thinking=a.thinking, cache_dir=a.cache_dir)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
