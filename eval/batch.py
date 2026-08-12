"""Shared plumbing for the vendors' 50%-discount batch APIs.

Batch trades latency for cost. Two consequences the paper has to state:

1. COST. Batched rows bill at input_per_1m_batch / output_per_1m_batch, half the
   synchronous rate. The rate is chosen per CACHED ENTRY from the "mode" recorded
   on that entry, never from a flag on the current run, so a cache holding a mix
   of synchronous smoke-test responses and batched full-split responses still
   prices each row at what was actually paid for it.

2. LATENCY. A batch response has no meaningful per-query latency, so batched rows
   carry latency_s = None. Every row in results/rows/*.json already has
   latency_s_mean = null, so batching costs us
   nothing we currently have, but the frontier rows cannot fill that column
   either. If the paper wants latency, run a small synchronous probe
   (--limit ~20, --cache-dir elsewhere) and report it as a probe, not as a
   full-split mean.

Job state lives in results/batch_jobs/<tag>-<split>.json so a submit and its
collect can be separated by hours, a crash, or a different session.
"""
import datetime, json, pathlib
from data.normalize import load_queries
from eval import protocol
from eval.pricing import cost_usd

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "results" / "batch_jobs"
CACHE_ROOT = ROOT / "results" / "baseline_cache"
PREDS_DIR = ROOT / "results" / "preds"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_tag(model, budget=None, thinking="off"):
    """Run name. A non-default budget or thinking regime is a DEVIATION and is
    tagged in, so a deviant run can never overwrite the protocol run's preds."""
    budget = budget or protocol.PROMPT_WORD_BUDGET
    tag = model
    if budget != protocol.PROMPT_WORD_BUDGET:
        tag += f"-w{budget}"
    if thinking != "off":
        tag += f"-think{thinking}"
    return tag


def cache_dir_for(tag):
    return CACHE_ROOT / tag


def pending(split, cache_dir, limit=None):
    """Queries with no cached response yet. Only these go into a batch, so a
    resubmit after a partial failure re-bills only what is actually missing."""
    cache_dir = pathlib.Path(cache_dir)
    return [q for q in load_queries(split)[:limit]
            if not (cache_dir / f"{q['query_id']}.json").exists()]


def preds_row(model, query_id, c, prices):
    """One line of a preds jsonl, priced at the rate the response was billed at."""
    out_toks = c["completion_tokens"] + (c.get("thoughts_tokens") or 0)
    mode = c.get("mode", "sync")
    return {"query_id": query_id,
            "prediction": c["prediction"],
            "cost_usd": cost_usd(model, c["prompt_tokens"], out_toks, prices,
                                 batch=(mode == "batch")),
            "latency_s": c.get("latency_s"),
            "model_snapshot": c.get("model_snapshot"),
            "called_at": c.get("called_at"),
            "mode": mode}


def write_preds(model, split, tag, prices, limit=None, cache_dir=None, preds_dir=None):
    """Materialise the preds jsonl from whatever is in the cache.

    Missing entries are skipped and counted rather than faked, so a partially
    collected batch produces a short file you can see is short.
    """
    cache_dir = pathlib.Path(cache_dir or cache_dir_for(tag))
    preds_dir = pathlib.Path(preds_dir or PREDS_DIR)
    preds_dir.mkdir(parents=True, exist_ok=True)
    out = preds_dir / f"baseline-{tag}-{split}.jsonl"
    written = missing = 0
    with out.open("w", encoding="utf-8") as f:
        for q in load_queries(split)[:limit]:
            cache = cache_dir / f"{q['query_id']}.json"
            if not cache.exists():
                missing += 1
                continue
            c = json.loads(cache.read_text(encoding="utf-8"))
            f.write(json.dumps(preds_row(model, q["query_id"], c, prices)) + "\n")
            written += 1
    return out, written, missing


# --- job state --------------------------------------------------------------

def job_path(tag, split, jobs_dir=None):
    return pathlib.Path(jobs_dir or JOBS_DIR) / f"{tag}-{split}.json"


def save_job(job, jobs_dir=None):
    p = job_path(job["tag"], job["split"], jobs_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return p


def load_job(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def list_jobs(jobs_dir=None):
    d = pathlib.Path(jobs_dir or JOBS_DIR)
    return sorted(d.glob("*.json")) if d.exists() else []
