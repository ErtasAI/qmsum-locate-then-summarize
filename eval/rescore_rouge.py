"""One-purpose migration: recompute ROUGE on every results row after the 2026-07-27
ROUGE-Lsum sentence-segmentation fix.

Kept in the repo rather than run as a throwaway, because it is the audit trail for a
change that touched every published number in results/rows/.

What it recomputes: rouge1, rouge2, rougeL, rougeLsum, from the stored predictions.
No model is re-run and no API is called; ROUGE is a scoring-time function of text
that is already on disk.

What it preserves: bertscore_f1 (embedding-based, does not use sentence segmentation,
and expensive to recompute), cost_usd_total, all latency fields, device, billing_mode
and notes.

Expected effect, verified before writing: rouge1/2/L are byte-identical everywhere
(newlines are whitespace to the n-gram tokenizer), and only rougeLsum moves. Any row
whose rouge1 changes indicates the predictions file no longer matches what was
scored, so the tool fails loudly rather than overwriting it.
"""
import argparse, json, pathlib

from data.normalize import load_queries
from eval.scorer import score

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTE = ("RLsum rescored 2026-07-27 under sentence segmentation "
        "(eval/scorer.py sentences()); R1/R2/RL unchanged.")


def rescore_row(row_path, refs_by_split, apply=False):
    d = json.loads(pathlib.Path(row_path).read_text(encoding="utf-8"))
    if "rouge1" not in d:
        return None
    pred = ROOT / "results" / "preds" / f"{d['run_id']}.jsonl"
    if not pred.exists():
        return {"run_id": d["run_id"], "status": "no preds file"}
    refs = refs_by_split[d["split"]]
    P = [json.loads(l) for l in pred.read_text(encoding="utf-8").splitlines() if l.strip()]
    P = [p for p in P if p["query_id"] in refs]
    s = score([p["prediction"] or "" for p in P], [refs[p["query_id"]] for p in P])

    drift = {k: (d[k], s[k]) for k in ("rouge1", "rouge2", "rougeL")
             if abs(d[k] - s[k]) > 1e-9}
    if drift:
        raise SystemExit(
            f"{d['run_id']}: rouge1/2/L changed, which segmentation cannot cause. "
            f"The predictions file no longer matches what was scored. {drift}")

    out = {"run_id": d["run_id"], "status": "ok", "n_old": d["n"], "n_new": len(P),
           "rougeLsum_old": d["rougeLsum"], "rougeLsum_new": s["rougeLsum"],
           "delta": s["rougeLsum"] - d["rougeLsum"]}
    if apply:
        d.update({k: s[k] for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")})
        if NOTE not in (d.get("notes") or ""):
            d["notes"] = ((d.get("notes") or "") + " | " + NOTE).strip(" |")
        pathlib.Path(row_path).write_text(json.dumps(d, indent=2), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the rows back; without it this is a dry run")
    a = ap.parse_args()
    refs = {s: {q["query_id"]: q["reference"] for q in load_queries(s)}
            for s in ("val", "test")}
    results = []
    for rp in sorted((ROOT / "results" / "rows").glob("*.json")):
        r = rescore_row(rp, refs, apply=a.apply)
        if r:
            results.append(r)
    print(f"{'run_id':36s} {'n':>5s} {'RLsum old':>9s} {'RLsum new':>9s} {'delta':>8s}")
    for r in results:
        if r["status"] != "ok":
            print(f"{r['run_id']:36s}  {r['status']}"); continue
        n = f"{r['n_new']}" + ("" if r["n_new"] == r["n_old"] else f"!={r['n_old']}")
        print(f"{r['run_id']:36s} {n:>5s} {r['rougeLsum_old']:9.4f} "
              f"{r['rougeLsum_new']:9.4f} {r['delta']:+8.4f}")
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{len(ok)} rows {'REWRITTEN' if a.apply else 'previewed (dry run; pass --apply)'}")


if __name__ == "__main__":
    main()
