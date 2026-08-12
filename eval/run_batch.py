"""Driver for the frontier baselines over the vendors' 50%-discount batch APIs.

    python -m eval.run_batch submit  --split test --models gpt-5.6-sol claude-opus-5
    python -m eval.run_batch status
    python -m eval.run_batch collect
    python -m eval.run_batch watch --interval 120

submit and collect are separable on purpose: a batch may take up to 24h, so the
job id is persisted to results/batch_jobs/ and collect can run in a later session.
`watch` is submit-agnostic; it polls whatever jobs exist, collects each as it ends,
and exits once none are outstanding. Run it in the background.

Collect is idempotent. Responses land in the per-query cache, and a resubmit only
bills the queries still missing from that cache.
"""
import argparse, json, pathlib, sys, time

from eval import batch as B
from eval import baselines_anthropic as ANTHROPIC
from eval import baselines_openai as OPENAI
from eval.pricing import load_prices

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROVIDERS = {"openai": OPENAI, "anthropic": ANTHROPIC}
DONE = {"openai": "completed", "anthropic": "ended"}


def provider_of(model):
    if model in OPENAI.MODELS:
        return "openai"
    if model in ANTHROPIC.MODELS:
        return "anthropic"
    raise SystemExit(f"unknown model {model!r}; not in baselines_openai.MODELS or "
                     f"baselines_anthropic.MODELS")


def clients(needed):
    """Only construct the clients actually required, so a missing key for a vendor
    you are not running is not a failure."""
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
    out = {}
    if "openai" in needed:
        from openai import OpenAI
        out["openai"] = OpenAI(timeout=1800.0)      # multi-MB batch uploads
    if "anthropic" in needed:
        import anthropic
        out["anthropic"] = anthropic.Anthropic(timeout=1800.0)
    return out


def cmd_submit(a):
    provs = {provider_of(m) for m in a.models}
    cl = clients(provs)
    for model in a.models:
        prov = provider_of(model)
        mod = PROVIDERS[prov]
        kw = dict(limit=a.limit, budget=a.budget)
        if prov == "anthropic":
            kw["thinking"] = a.thinking
        job = mod.submit_batch(cl[prov], model, a.split, **kw)
        if job is None:
            print(f"[{model}] cache already complete for {a.split}, nothing to submit")
            continue
        print(f"[{model}] submitted {job['n_requests']} requests "
              f"({job['payload_mb']} MB) as {job['batch_id']}")


def _jobs():
    return [B.load_job(p) for p in B.list_jobs()]


def cmd_status(a):
    jobs = _jobs()
    if not jobs:
        print("no batch jobs on record")
        return
    cl = clients({j["provider"] for j in jobs})
    for job in jobs:
        st = PROVIDERS[job["provider"]].poll_batch(cl[job["provider"]], job)
        cached = len(list(pathlib.Path(job["cache_dir"]).glob("*.json")))
        print(f"{job['tag']:22s} {job['split']:5s} {st['status']:12s} "
              f"done={st['completed']}/{st['total']} failed={st['failed']} cached={cached}")


def _collect_one(cl, job, prices):
    mod = PROVIDERS[job["provider"]]
    state, written, errors = mod.collect_batch(cl[job["provider"]], job)
    if state["status"] != DONE[job["provider"]]:
        return state, None
    out, n, missing = B.write_preds(job["model"], job["split"], job["tag"], prices,
                                    limit=job.get("limit"))
    total = sum(json.loads(l)["cost_usd"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"[{job['tag']}] cached {written} new, wrote {n} preds "
          f"({missing} missing) -> {out}")
    print(f"[{job['tag']}] batch-rate cost for this split: ${total:.2f}")
    if errors:
        print(f"[{job['tag']}] {len(errors)} FAILED, not cached: {errors[:5]}")
    if job["provider"] == "anthropic":
        leaked = ANTHROPIC.scan_leaks(out)
        print(f"[{job['tag']}] leak scan: "
              + (f"LEAKED in {len(leaked)}: {leaked[:5]} -- do NOT score" if leaked else "clean"))
    return state, out


def cmd_collect(a):
    jobs = _jobs()
    if not jobs:
        print("no batch jobs on record")
        return
    cl = clients({j["provider"] for j in jobs})
    prices = load_prices()
    for job in jobs:
        state, out = _collect_one(cl, job, prices)
        if out is None:
            print(f"[{job['tag']}] not ready: {state['status']} "
                  f"({state['completed']}/{state['total']})")


def cmd_watch(a):
    """Poll until every job has ended, collecting each as it does. One line per
    state change so a background run is readable, plus failure states, not just
    the happy path."""
    prices = load_prices()
    outstanding = {B.job_path(j["tag"], j["split"]).name: j for j in _jobs()}
    if not outstanding:
        print("no batch jobs on record"); return
    cl = clients({j["provider"] for j in outstanding.values()})
    deadline = time.time() + a.max_hours * 3600
    last = {}
    while outstanding and time.time() < deadline:
        for name, job in list(outstanding.items()):
            prov = job["provider"]
            try:
                st = PROVIDERS[prov].poll_batch(cl[prov], job)
            except Exception as e:                      # transient API error, keep watching
                print(f"[{job['tag']}] poll error: {type(e).__name__}: {e}")
                continue
            key = (st["status"], st["completed"])
            if last.get(name) != key:
                print(f"[{job['tag']}] {st['status']} {st['completed']}/{st['total']} "
                      f"failed={st['failed']}")
                last[name] = key
            if st["status"] in ("cancelled", "canceled", "failed", "expired"):
                print(f"[{job['tag']}] TERMINAL FAILURE: {st['status']}")
                outstanding.pop(name); continue
            if st["status"] == DONE[prov]:
                _collect_one(cl, job, prices)
                outstanding.pop(name)
        if outstanding:
            time.sleep(a.interval)
    if outstanding:
        print(f"TIMEOUT after {a.max_hours}h, still outstanding: "
              f"{[j['tag'] for j in outstanding.values()]}")
        sys.exit(1)
    print("all batches collected")


def cmd_pipeline(a):
    """submit -> wait -> collect, looping until each model's split is complete.

    Needed because OpenAI caps ENQUEUED tokens per model (900k on gpt-5.6-sol for
    this org) and one QMSum split is ~3.5M, so the split cannot be enqueued at once;
    a chunk must finish before the next is accepted. Anthropic has no such cap, so
    its models complete on the first pass. Chunk boundaries are invisible in the
    output: every response lands in the same per-query cache and the preds file is
    rewritten whole each round.
    """
    prices = load_prices()
    provs = {provider_of(m) for m in a.models}
    cl = clients(provs)
    remaining = list(a.models)
    rounds = 0
    while remaining and rounds < a.max_rounds:
        rounds += 1
        active = []
        for model in list(remaining):
            prov = provider_of(model)
            kw = dict(limit=a.limit, budget=a.budget)
            if prov == "anthropic":
                kw["thinking"] = a.thinking
            job = PROVIDERS[prov].submit_batch(cl[prov], model, a.split, **kw)
            if job is None:
                print(f"[{model}] split complete")
                remaining.remove(model)
                continue
            print(f"[{model}] round {rounds}: enqueued {job['n_requests']} "
                  f"(~{job.get('est_enqueued_tokens', 0):,} tok), "
                  f"{job.get('remaining', 0)} left after this -> {job['batch_id']}")
            active.append(job)
        if not active:
            break
        pending = {j["tag"]: j for j in active}
        while pending:
            time.sleep(a.interval)
            for tag, job in list(pending.items()):
                prov = job["provider"]
                try:
                    st = PROVIDERS[prov].poll_batch(cl[prov], job)
                except Exception as e:
                    print(f"[{tag}] poll error: {type(e).__name__}: {e}")
                    continue
                if st["status"] in ("cancelled", "canceled", "failed", "expired"):
                    print(f"[{tag}] TERMINAL FAILURE: {st['status']}")
                    _explain_failure(cl[prov], job)
                    pending.pop(tag)
                    if job["model"] in remaining:
                        remaining.remove(job["model"])
                    continue
                if st["status"] == DONE[prov]:
                    _collect_one(cl, job, prices)
                    pending.pop(tag)
    if remaining:
        print(f"stopped with models incomplete: {remaining}")
        sys.exit(1)
    print("all splits complete")


def _explain_failure(client, job):
    """Surface the vendor's own reason. A silent 'failed' costs a debugging cycle."""
    try:
        if job["provider"] == "openai":
            b = client.batches.retrieve(job["batch_id"])
            for e in (getattr(b.errors, "data", None) or [])[:3]:
                print(f"    {e.code}: {e.message}")
    except Exception as e:
        print(f"    (could not retrieve failure detail: {type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--split", required=True, choices=["val", "test"])
    s.add_argument("--limit", type=int)
    s.add_argument("--budget", type=int, default=None)
    s.add_argument("--thinking", choices=["off", "low"], default="off")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("collect"); s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("watch")
    s.add_argument("--interval", type=int, default=120)
    s.add_argument("--max-hours", type=float, default=25.0)
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("pipeline")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--split", required=True, choices=["val", "test"])
    s.add_argument("--limit", type=int)
    s.add_argument("--budget", type=int, default=None)
    s.add_argument("--thinking", choices=["off", "low"], default="off")
    s.add_argument("--interval", type=int, default=60)
    s.add_argument("--max-rounds", type=int, default=12)
    s.set_defaults(fn=cmd_pipeline)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
