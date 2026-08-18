"""One-purpose migration: compute the BERTScore that three M0 community-checkpoint rows
never got.

Why they were blank. The four community rows come from `eval/hf_rows.py`, which handles each
checkpoint's native `query </s> transcript` input format and carries no `--bertscore` flag to
pass; BERTScore is opt-in in `eval/run_eval.py` and `eval/scorer.py` returns
`bertscore_f1: None` without it. So the M0 pass of 2026-07-23 recorded all four community rows
as "not computed this pass". distilbart was picked up on 2026-07-31 by the per-query interval
pass that closed the BERTScore half of the frontier claim, because it is the strongest community
checkpoint and the only one that claim binds on. bart-large-cnn, pegasus and led-base sat 8.1,
15.3 and 26.0 ROUGE-1 below the promoted row, no claim turned on them, and they stayed blank.

Kept in the repo rather than run as a throwaway, for the same reason `eval/rescore_rouge.py` is:
it is the audit trail for a change that writes to `results/rows/`.

What it computes: `bertscore_f1`, from the stored predictions, through `eval/scorer.py` so the
call is byte-identical to the one every other row's figure came from. No model is re-run and no
API is called. BERTScore is a scoring-time function of text already on disk.

What it preserves: rouge1, rouge2, rougeL, rougeLsum, cost_usd_total, every latency and VRAM
field, and notes.

Two guards, both fail loudly:

1. **ROUGE must reproduce byte-identically.** Rescoring the stored predictions has to return the
   published ROUGE. A drift means the predictions file no longer matches what was scored, and the
   row must not be touched until that is understood.
2. **A row that already has a BERTScore is treated as a control, never as a target.** Its stored
   value must reproduce. `--only row-distilbart-test` is therefore a live check that this tool
   agrees with the 2026-07-31 pass, and it passes at 0.854595424017448.

Usage:
    python -m eval.backfill_bertscore --only row-bart-large-cnn-test,row-pegasus-test --apply
    python -m eval.backfill_bertscore --missing            # dry run over every blank row
"""
import argparse, json, pathlib

from data.normalize import load_queries
from eval.scorer import score

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTE = ("bertscore_f1 backfilled 2026-08-17 from the stored predictions "
        "(eval/backfill_bertscore.py); ROUGE reproduced byte-identically and is unchanged.")
ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def backfill_row(row_path, refs_by_split, apply=False):
    d = json.loads(pathlib.Path(row_path).read_text(encoding="utf-8"))
    if "rouge1" not in d:
        return None
    pred = ROOT / "results" / "preds" / f"{d['run_id']}.jsonl"
    if not pred.exists():
        return {"run_id": d["run_id"], "status": "no preds file"}

    refs = refs_by_split[d["split"]]
    P = [json.loads(l) for l in pred.read_text(encoding="utf-8").splitlines() if l.strip()]
    P = [p for p in P if p["query_id"] in refs]
    s = score([p["prediction"] or "" for p in P], [refs[p["query_id"]] for p in P],
              with_bertscore=True)

    drift = {k: (d[k], s[k]) for k in ROUGE_KEYS if abs(d[k] - s[k]) > 1e-9}
    if drift:
        raise SystemExit(
            f"{d['run_id']}: ROUGE moved on a rescore of the same predictions, so the "
            f"predictions file no longer matches what was scored. Refusing to write. {drift}")

    old = d.get("bertscore_f1")
    if old is not None:
        if abs(old - s["bertscore_f1"]) > 1e-9:
            raise SystemExit(
                f"{d['run_id']}: stored bertscore_f1 {old!r} does not reproduce "
                f"({s['bertscore_f1']!r}). Refusing to write.")
        return {"run_id": d["run_id"], "status": "control ok", "n": len(P),
                "bertscore_old": old, "bertscore_new": s["bertscore_f1"]}

    out = {"run_id": d["run_id"], "status": "backfilled" if apply else "would backfill",
           "n": len(P), "bertscore_old": None, "bertscore_new": s["bertscore_f1"]}
    if apply:
        d["bertscore_f1"] = s["bertscore_f1"]
        if NOTE not in (d.get("notes") or ""):
            d["notes"] = ((d.get("notes") or "") + " | " + NOTE).strip(" |")
        # Trailing newline: every committed row file has one, and dropping it puts a
        # "\ No newline at end of file" marker in the diff of a change that moved one field.
        pathlib.Path(row_path).write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--only", help="comma-separated run_ids")
    g.add_argument("--missing", action="store_true",
                   help="every row whose bertscore_f1 is null")
    ap.add_argument("--apply", action="store_true",
                    help="write the rows back; without it this is a dry run")
    a = ap.parse_args()

    wanted = set(a.only.split(",")) if a.only else None
    refs = {s: {q["query_id"]: q["reference"] for q in load_queries(s)}
            for s in ("val", "test")}

    targets = []
    for rp in sorted((ROOT / "results" / "rows").glob("*.json")):
        d = json.loads(rp.read_text(encoding="utf-8"))
        if wanted is not None:
            if d.get("run_id") in wanted:
                targets.append(rp)
        elif d.get("bertscore_f1") is None:
            targets.append(rp)

    if wanted is not None:
        found = {json.loads(p.read_text(encoding="utf-8"))["run_id"] for p in targets}
        if missing := wanted - found:
            raise SystemExit(f"no row file for: {sorted(missing)}")

    results = [r for rp in targets if (r := backfill_row(rp, refs, apply=a.apply))]
    print(f"{'run_id':36s} {'n':>5s} {'BERTScore':>10s}  status")
    for r in results:
        if r["status"] == "no preds file":
            print(f"{r['run_id']:36s} {'':>5s} {'':>10s}  no preds file"); continue
        print(f"{r['run_id']:36s} {r['n']:5d} {r['bertscore_new']:10.6f}  {r['status']}")
    written = [r for r in results if r["status"] == "backfilled"]
    controls = [r for r in results if r["status"] == "control ok"]
    print(f"\n{len(written)} rows written, {len(controls)} controls reproduced"
          + ("" if a.apply else "  (dry run; pass --apply)"))


if __name__ == "__main__":
    main()
