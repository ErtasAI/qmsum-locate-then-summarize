"""Blind pairwise judge.

Blinding (unchanged, and regression-tested):
  - systems are presented as anonymous "Summary A" / "Summary B"
  - no model name, vendor, or snapshot ID ever enters the prompt
  - position is swapped on alternating items so A/B order cannot carry signal

SELF-PREFERENCE. A judge cannot be trusted to score its own vendor's output.
Blinding does not fix this, because models recognise their own output
distribution. run_judgement() detects a vendor clash between judge and
contestant and records it as self_preference_risk. The panel driver
(eval/run_judge_panel.py) exploits this rather than merely warning about it: it
runs BOTH vendors' judges over every pair, so each pair carries one clean
cross-vendor verdict and one same-vendor verdict, and the difference between
them is a measurement of the bias instead of a caveat about it.

PROTOCOL KNOBS. The frozen protocol's `temperature=0` is a 400 on GPT-5.6 and
Claude 5 (measured 2026-07-27). request_kwargs_* below
encode what each model actually accepts; the shapes are asserted in tests so
they cannot be quietly reintroduced. Reasoning is disabled on the judges that
allow it: the prompt asks for written reasoning before the verdict, and visible
reasoning is auditable in a way a hidden trace is not.

JUDGE TOKEN BUDGET. Judges get JUDGE_MAX_TOKENS, not protocol.MAX_NEW_TOKENS.
A judge is not a scored generation, and a truncated judgement loses its VERDICT
line and reads as UNPARSED. Truncation is recorded per response so an unparsed
verdict can never be mistaken for an ambivalent one.
"""
import argparse, datetime, json, pathlib, random, time
from eval import protocol
from eval.pricing import cost_usd, load_prices

ROOT = pathlib.Path(__file__).resolve().parents[1]

JUDGE_MAX_TOKENS = 1024

# Transient API failures. A judge panel is hundreds of sequential synchronous
# calls, so the chance of hitting at least one of these approaches certainty:
# an unretried 529 killed a full preference-panel run on 2026-07-27 after one
# completed leg. Matched on status code and on exception class name so this
# stays vendor-agnostic and testable without importing either SDK.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRY_NAMES = ("overloaded", "ratelimit", "apiconnection", "apitimeout",
               "internalserver", "serviceunavailable", "timeout")
RETRY_ATTEMPTS = 6
RETRY_BASE_DELAY = 2.0


def is_transient(exc):
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in RETRY_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(m in name for m in RETRY_NAMES)


def call_with_retry(fn, *args, attempts=RETRY_ATTEMPTS, base_delay=RETRY_BASE_DELAY,
                    sleep=time.sleep, rng=None, **kwargs):
    """Retry transient API failures with exponential backoff and jitter.

    A non-transient error is re-raised immediately: a 400 on a bad request body
    is a bug to fix, and burning six attempts on it would only hide it.
    """
    rng = rng or random.Random()
    last = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:            # noqa: BLE001 - re-raised below
            if not is_transient(exc) or attempt == attempts - 1:
                raise
            last = exc
            sleep(base_delay * (2 ** attempt) * (0.5 + rng.random()))
    raise last  # unreachable; the loop either returns or raises

DEFAULT_JUDGE_MODEL = {"gpt": "gpt-5.6-sol", "claude": "claude-opus-5",
                       "gemini": "gemini-3.6-flash"}
VENDOR_PREFIXES = {"gpt": "openai", "o1": "openai", "o3": "openai",
                   "claude": "anthropic", "gemini": "google"}

# Models that 400 on `temperature`, and models that reason unless told not to.
# Mirrors baselines_openai.py / baselines_anthropic.py; see the protocol-limit note
# in the module docstring.
NO_SAMPLING_PARAMS = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
                      "claude-opus-5", "claude-sonnet-5"}
GPT_REASONS_BY_DEFAULT = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
CLAUDE_THINKING_ON_BY_DEFAULT = {"claude-opus-5", "claude-sonnet-5"}


def vendor_of(model_id):
    if not model_id: return None
    for prefix, vendor in VENDOR_PREFIXES.items():
        if model_id.lower().startswith(prefix): return vendor
    return None

def vendor_of_preds(pred_file):
    """Vendor that produced a predictions file, from its recorded snapshot."""
    for line in pathlib.Path(pred_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            return vendor_of(json.loads(line).get("model_snapshot"))
    return None

def build_judge_prompt(query, summary_a, summary_b, reference=None):
    """Two judging modes. Both are reported; the gap between them is the result.

    PREFERENCE (reference=None) asks which summary is better, full stop. This is
    the conventional LLM-as-judge setup and it carries a known verbosity bias:
    with no length anchor, a longer summary looks more comprehensive. That bias
    is not hypothetical on this panel. Measured median lengths against a 59-word
    gold reference: ours 56 words (0.95x), GPT rows 81-86 (~1.4x), Claude rows
    190-223 (3.2-3.8x). A preference judge therefore partly measures length.

    GROUNDED (reference given) asks which summary better matches the human
    reference answer. This is the mode that speaks to the actual claim, because
    the task is "reproduce what the annotator wrote", and it is what ROUGE
    measures without ROUGE's n-gram literalism. Running both turns the verbosity
    confound into a measured quantity instead of a caveat, exactly as running
    both vendors' judges does for self-preference.
    """
    if reference is None:
        return (
            "Two anonymous systems summarized the same meeting to answer a query. "
            "Judge which summary answers the query better on faithfulness, coverage, "
            "and concision. Reply with reasoning, then a final line 'VERDICT: A', "
            "'VERDICT: B', or 'VERDICT: TIE'.\n\n"
            f"Query: {query}\n\nSummary A:\n{summary_a}\n\nSummary B:\n{summary_b}")
    return (
        "Two anonymous systems summarized the same meeting to answer a query. A "
        "human-written reference answer is given. Judge which summary better "
        "matches the reference in content and emphasis: the facts it selects, "
        "what it treats as important, and its level of detail. Extra material "
        "not in the reference is not a merit. Reply with reasoning, then a final "
        "line 'VERDICT: A', 'VERDICT: B', or 'VERDICT: TIE'.\n\n"
        f"Query: {query}\n\nReference answer:\n{reference}\n\n"
        f"Summary A:\n{summary_a}\n\nSummary B:\n{summary_b}")

def parse_verdict(text):
    for line in reversed((text or "").strip().splitlines()):
        if line.strip().startswith("VERDICT:"):
            return line.split(":", 1)[1].strip().upper()
    return "UNPARSED"

def pick_sample(query_ids, n, seed=protocol.SEED):
    return sorted(random.Random(seed).sample(sorted(query_ids), n))


# --- per-vendor call shapes -------------------------------------------------

def request_kwargs_gpt(model):
    kw = {}
    if model in GPT_REASONS_BY_DEFAULT:
        kw["reasoning_effort"] = "none"
    if model not in NO_SAMPLING_PARAMS:
        kw["temperature"] = protocol.TEMPERATURE
    return kw

def request_kwargs_claude(model):
    kw = {"max_tokens": JUDGE_MAX_TOKENS}
    if model in CLAUDE_THINKING_ON_BY_DEFAULT:
        kw["thinking"] = {"type": "disabled"}
    if model not in NO_SAMPLING_PARAMS:
        kw["temperature"] = protocol.TEMPERATURE
    return kw

def _usage(obj, *names):
    """Token count from a response, or None when the object does not carry usage.

    None, never 0: an absent count must not read as a free API call downstream.
    """
    u = getattr(obj, "usage", None)
    if u is None: return None
    for n in names:
        v = getattr(u, n, None)
        if v is not None: return v
    return None

def _gpt_judge(client, model, prompt, max_tokens=None):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens or JUDGE_MAX_TOKENS,
        **request_kwargs_gpt(model))
    choice = r.choices[0]
    return {"text": choice.message.content,
            "snapshot": r.model,
            "prompt_tokens": _usage(r, "prompt_tokens"),
            "completion_tokens": _usage(r, "completion_tokens"),
            "truncated": getattr(choice, "finish_reason", None) == "length"}

def _claude_judge(client, model, prompt, max_tokens=None):
    kw = request_kwargs_claude(model)
    if max_tokens: kw["max_tokens"] = max_tokens
    r = client.messages.create(
        model=model, messages=[{"role": "user", "content": prompt}], **kw)
    return {"text": "".join(b.text for b in r.content if b.type == "text"),
            "snapshot": r.model,
            "prompt_tokens": _usage(r, "input_tokens"),
            "completion_tokens": _usage(r, "output_tokens"),
            "truncated": getattr(r, "stop_reason", None) == "max_tokens"}

def _gemini_judge(client, model, prompt, max_tokens=None):
    r = client.models.generate_content(model=model, contents=prompt)
    return {"text": r.text, "snapshot": getattr(r, "model_version", model),
            "prompt_tokens": None, "completion_tokens": None, "truncated": False}

JUDGES = {"gpt": _gpt_judge, "claude": _claude_judge, "gemini": _gemini_judge}


# --- caching ----------------------------------------------------------------

def cache_key(qid, swap):
    """Orientation is part of the key, not a field to be validated on read.

    A response is an answer to one specific A/B presentation. Changing n
    reshuffles the sample and can flip an item's parity; keying on orientation
    means that adds a new entry instead of silently reusing the wrong one.
    """
    return f"{qid}-{'ba' if swap else 'ab'}.json"


def run_judgement(pred_a_file, pred_b_file, split, n, client, provider="gpt",
                  model=None, cache_dir=None, prices=None, mode="preference"):
    """Judge A against B on n sampled queries. A is ours by convention.

    mode="preference" judges the summaries alone; mode="grounded" shows the
    judge the human reference answer. See build_judge_prompt for why both run.

    cache_dir=None means no caching (tests). The driver always passes one:
    600 judge calls with no cache re-bills from zero after any crash.
    """
    from data.normalize import load_queries
    if mode not in ("preference", "grounded"):
        raise ValueError(f"unknown judge mode {mode!r}")
    model = model or DEFAULT_JUDGE_MODEL[provider]
    call = JUDGES[provider]
    prices = prices if prices is not None else load_prices()
    cache_dir = pathlib.Path(cache_dir) if cache_dir else None
    if cache_dir: cache_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_queries(split)
    queries = {q["query_id"]: q["query"] for q in loaded}
    references = {q["query_id"]: q["reference"] for q in loaded}
    a = {json.loads(l)["query_id"]: json.loads(l)["prediction"] for l in open(pred_a_file, encoding="utf-8")}
    b = {json.loads(l)["query_id"]: json.loads(l)["prediction"] for l in open(pred_b_file, encoding="utf-8")}
    sample = pick_sample(set(a) & set(b), n)

    judge_vendor = vendor_of(model)
    contestant_vendors = {v for v in (vendor_of_preds(pred_a_file), vendor_of_preds(pred_b_file)) if v}
    tally = {"a_wins": 0, "b_wins": 0, "ties": 0, "unparsed": 0, "truncated": 0,
             "n_sampled": len(sample), "mode": mode,
             "judge_provider": provider, "judge_model": model,
             "judge_vendor": judge_vendor,
             "contestant_vendors": sorted(contestant_vendors),
             "self_preference_risk": judge_vendor in contestant_vendors,
             "judged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "judge_snapshot": None, "cost_usd": 0.0, "entries_without_usage": 0,
             "sample_ids": sample, "items": []}

    for i, qid in enumerate(sample):
        swap = i % 2 == 1  # position swap on alternating items
        cache_file = cache_dir / cache_key(qid, swap) if cache_dir else None
        if cache_file and cache_file.exists():
            resp = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            first, second = (b, a) if swap else (a, b)
            resp = call_with_retry(call, client, model, build_judge_prompt(
                queries[qid], first[qid], second[qid],
                reference=references[qid] if mode == "grounded" else None))
            resp["called_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if cache_file:
                cache_file.write_text(json.dumps(resp), encoding="utf-8")

        tally["judge_snapshot"] = resp["snapshot"]
        if resp.get("truncated"): tally["truncated"] += 1
        if resp.get("prompt_tokens") is None or resp.get("completion_tokens") is None:
            tally["entries_without_usage"] += 1
        else:
            tally["cost_usd"] += cost_usd(model, resp["prompt_tokens"],
                                          resp["completion_tokens"], prices)

        v = parse_verdict(resp["text"])
        # UNPARSED is its own bucket. Folding it into ties would let a truncated
        # or malformed response read as a considered draw.
        if v == "UNPARSED":
            tally["unparsed"] += 1
            winner = None
        elif v == "TIE":
            tally["ties"] += 1
            winner = "tie"
        elif (v == "A") != swap:
            tally["a_wins"] += 1
            winner = "a"
        else:
            tally["b_wins"] += 1
            winner = "b"
        tally["items"].append({"query_id": qid, "swap": swap, "verdict": v,
                               "winner": winner, "truncated": bool(resp.get("truncated"))})

    tally["cost_usd"] = round(tally["cost_usd"], 4)
    tally["n_decided"] = tally["a_wins"] + tally["b_wins"] + tally["ties"]
    return tally


def main():
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True); ap.add_argument("--theirs", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--judge", choices=["gpt", "claude", "gemini"], default="gpt")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--n", type=int, default=protocol.JUDGE_SAMPLE_N)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--mode", choices=["preference", "grounded"], default="preference",
                    help="preference = summaries alone (carries verbosity bias); "
                         "grounded = judge also sees the human reference answer")
    a = ap.parse_args()
    if a.judge == "gpt":
        from openai import OpenAI; client = OpenAI()
    elif a.judge == "claude":
        import anthropic; client = anthropic.Anthropic()
    else:
        from google import genai; client = genai.Client()
    tally = run_judgement(a.ours, a.theirs, a.split, a.n, client,
                          provider=a.judge, model=a.judge_model, cache_dir=a.cache_dir,
                          mode=a.mode)
    pathlib.Path(a.out).write_text(json.dumps(tally, indent=2))
    print(json.dumps({k: v for k, v in tally.items()
                      if k not in ("sample_ids", "items")}, indent=2))
    if tally["self_preference_risk"]:
        print(f"\nWARNING: the judge ({tally['judge_vendor']}) shares a vendor with a "
              f"contestant {tally['contestant_vendors']}. Blinding does not remove "
              f"self-preference bias. Re-judge this pair with a different vendor before "
              f"using the number.")

if __name__ == "__main__":
    main()
