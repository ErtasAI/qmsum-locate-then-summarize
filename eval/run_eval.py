"""Score a prediction file against a split under the frozen protocol."""
import argparse, json, pathlib
from collections import Counter
from data.normalize import load_queries
from eval.scorer import score
from eval import vram

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="smoke runs only; full val/test rows must be complete")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    refs = {q["query_id"]: q["reference"] for q in load_queries(a.split)}
    preds = [json.loads(l) for l in open(a.pred, encoding="utf-8")]
    dupes = sorted(q for q, c in Counter(p["query_id"] for p in preds).items() if c > 1)
    if dupes:
        raise SystemExit(f"{len(dupes)} duplicate query_ids in predictions "
                         f"(e.g. {dupes[:3]}); a resumed run must dedupe before scoring")
    missing = set(refs) - {p["query_id"] for p in preds}
    if a.split in ("val", "test") and missing and not a.allow_partial:
        raise SystemExit(f"{len(missing)} queries missing predictions; partial rows are not scoreable")
    scored = [p for p in preds if p["query_id"] in refs]
    pairs = [(p["prediction"], refs[p["query_id"]]) for p in scored]
    if not pairs:
        raise SystemExit("no predictions matched the split's query_ids; nothing to score")

    s = score([p for p, _ in pairs], [r for _, r in pairs], with_bertscore=a.bertscore)
    costs = [p["cost_usd"] for p in scored if p.get("cost_usd") is not None]
    # Batched rows carry latency_s = None deliberately (a batch response has no
    # meaningful per-query latency), so filter on the VALUE, not on key presence,
    # or every batched row crashes here summing None.
    # The warmup query carries CUDA context creation and kernel autotuning and runs
    # several times slower than steady state; averaging it in would overstate latency.
    timed = [p for p in scored if p.get("latency_s") is not None and not p.get("warmup")]
    mean = lambda k: (round(sum(p[k] for p in timed) / len(timed), 3)
                      if timed and all(p.get(k) is not None for p in timed) else None)
    modes = sorted({p["mode"] for p in scored if p.get("mode")})
    devices = sorted({p["device"] for p in scored if p.get("device")})
    # Peak VRAM is a MAXIMUM over the workload, so unlike latency the warmup query is kept:
    # a card has to hold the largest query whenever it arrives, and memory carries no
    # first-call penalty of the kind that makes the first latency reading unusable.
    peak = vram.summarize([p.get("peak_vram_gb") for p in scored])
    peak_reserved = vram.summarize([p.get("peak_vram_reserved_gb") for p in scored])
    row = {"run_id": a.run_id, "split": a.split, "n": len(pairs), **s,
           "cost_usd_total": round(sum(costs), 6) if costs else None,
           "latency_s_mean": mean("latency_s"),
           "locate_s_mean": mean("locate_s"),
           "generate_s_mean": mean("generate_s"),
           "latency_n_timed": len(timed) or None,
           "peak_vram_gb_max": peak["max"] if peak else None,
           "peak_vram_gb_median": peak["median"] if peak else None,
           "peak_vram_gb_p95": peak["p95"] if peak else None,
           "peak_vram_reserved_gb_max": peak_reserved["max"] if peak_reserved else None,
           "device": "+".join(devices) or None,
           "billing_mode": "+".join(modes) or None,
           "notes": a.notes}
    out = ROOT / "results" / "rows" / f"{a.run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))

if __name__ == "__main__":
    main()
